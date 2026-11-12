//! Local package-v2 admission materialization. Network acquisition is caller-owned.
use crate::{source_registry::VerifiedFileFingerprint, stage_load::ResolvedStagePackage};
use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use skippy_package_format::{
    Artifact as PackageV2Artifact, PackageManifest as PackageManifestV2,
    stage_admission::StageAdmissionDescriptor as PackageV2StageAdmissionDescriptor,
};
use std::{
    collections::HashMap,
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
    sync::{Mutex, MutexGuard, OnceLock, PoisonError},
    time::{Duration, SystemTime},
};

pub fn ensure_package_manifest_sha(package_ref: &str, expected_sha256: &str) -> Result<()> {
    if expected_sha256.trim().is_empty() {
        return Ok(());
    }
    anyhow::ensure!(
        expected_sha256.len() == 64 && expected_sha256.chars().all(|ch| ch.is_ascii_hexdigit()),
        "package manifest sha256 must be a hex SHA-256 digest"
    );
    let manifest_path = Path::new(package_ref).join("model-package.json");
    let manifest_contents = fs::read(&manifest_path).context("read package manifest")?;
    let actual_sha = hex::encode(Sha256::digest(&manifest_contents));
    anyhow::ensure!(
        actual_sha.eq_ignore_ascii_case(expected_sha256),
        "package manifest sha256 mismatch"
    );
    Ok(())
}

pub fn verify_package_v2_artifact(package_dir: &Path, artifact: &PackageV2Artifact) -> Result<()> {
    let relative = safe_manifest_file_path(&artifact.path)?;
    let path = package_dir.join(&relative);
    let metadata = fs::metadata(&path)
        .with_context(|| format!("stat package-v2 artifact {}", relative.display()))?;
    anyhow::ensure!(
        metadata.is_file(),
        "package-v2 artifact is not a file: {}",
        relative.display()
    );
    anyhow::ensure!(
        metadata.len() == artifact.byte_size,
        "package-v2 artifact {} size differs from manifest",
        relative.display()
    );
    let matches = file_matches_sha256(&path, &artifact.sha256)
        .with_context(|| format!("hash package-v2 artifact {}", relative.display()))?;
    anyhow::ensure!(
        matches,
        "package-v2 artifact {} SHA-256 differs from manifest",
        relative.display()
    );
    Ok(())
}

/// A verified hash is reused, and recorded, only for a file whose last change
/// is at least this old when verification starts, so an edit within one
/// timestamp tick of the recorded change cannot hide behind an unchanged
/// fingerprint (git's "racily clean" rule, as in the native runtime cache).
const SETTLE_WINDOW: Duration = Duration::from_secs(2);

/// Whether the file at `path` has SHA-256 `expected_sha256`.
///
/// A stage load verifies the same package artifacts several times (artifact
/// transfer, acquisition, then admission), and a later load of the same
/// package verifies them again. Each artifact can be gigabytes, so this
/// process remembers every file it hashed to its expected digest together
/// with the file's fingerprint (size, mtime, change time, device, inode),
/// taken before hashing. A later check of the same file and digest skips the
/// read while the fingerprint is unchanged; any change hashes it again, so a
/// replaced or corrupted artifact is still rejected. Links are resolved first,
/// so Hugging Face snapshot entries are fingerprinted by the blob they name.
pub fn file_matches_sha256(path: &Path, expected_sha256: &str) -> Result<bool> {
    let started = verification_clock();
    let path = fs::canonicalize(path)?;
    let fingerprint =
        crate::source_registry::verified_path_fingerprint(std::slice::from_ref(&path))
            .and_then(|mut fingerprints| fingerprints.pop())
            .filter(|fingerprint| fingerprint.settled_at(started, SETTLE_WINDOW));
    let key = (path, expected_sha256.to_ascii_lowercase());
    if fingerprint.is_some() && verified_files().get(&key) == fingerprint.as_ref() {
        return Ok(true);
    }
    let matches = file_sha256(&key.0)? == key.1;
    // A mismatch, or a file too recently changed to trust, forgets any older
    // entry so the next check hashes again.
    let mut verified = verified_files();
    match fingerprint.filter(|_| matches) {
        Some(fingerprint) => {
            verified.insert(key, fingerprint);
        }
        None => {
            verified.remove(&key);
        }
    }
    Ok(matches)
}

/// Files this process already hashed to an expected digest, keyed by
/// canonical path and lowercase digest.
fn verified_files() -> MutexGuard<'static, HashMap<(PathBuf, String), VerifiedFileFingerprint>> {
    static VERIFIED: OnceLock<Mutex<HashMap<(PathBuf, String), VerifiedFileFingerprint>>> =
        OnceLock::new();
    VERIFIED
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// The time a verification starts.
fn verification_clock() -> SystemTime {
    #[cfg(test)]
    {
        SystemTime::now() + tests::CLOCK_AHEAD.with(std::cell::Cell::get)
    }
    #[cfg(not(test))]
    {
        SystemTime::now()
    }
}

fn file_sha256(path: &Path) -> Result<String> {
    #[cfg(test)]
    tests::HASHED.with(|count| count.set(count.get() + 1));
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hex::encode(hasher.finalize()))
}

pub fn safe_manifest_file_path(path: &str) -> Result<PathBuf> {
    anyhow::ensure!(!path.is_empty(), "manifest file path is empty");
    let path = Path::new(path);
    let mut components = path.components();
    let Some(first) = components.next() else {
        bail!("manifest file path is empty");
    };
    anyhow::ensure!(
        matches!(first, Component::Normal(_))
            && components.all(|component| matches!(component, Component::Normal(_))),
        "manifest file path must be a safe relative path: {}",
        path.display()
    );
    Ok(path.to_path_buf())
}

pub fn resolve_local_package_stage(
    package_dir: &Path,
    admission: &PackageV2StageAdmissionDescriptor,
) -> Result<(String, Vec<PathBuf>, Option<PathBuf>)> {
    let local_ref = package_dir.to_string_lossy().into_owned();
    let manifest_path = package_dir.join("model-package.json");
    let manifest_bytes = fs::read(&manifest_path)
        .with_context(|| format!("read package-v2 manifest {}", manifest_path.display()))?;
    let manifest: PackageManifestV2 =
        serde_json::from_slice(&manifest_bytes).context("parse package-v2 manifest")?;
    let manifest =
        skippy_model::package_carrier::resolve_package_carrier_from_dir(manifest, package_dir)
            .context("resolve package-v2 metadata carrier")?;
    let resolved = manifest
        .resolve_stage_admission(admission)
        .context("resolve exact package-v2 stage admission")?;
    let required = resolved
        .required_artifacts
        .iter()
        .map(|artifact| (*artifact).clone())
        .collect::<Vec<_>>();

    for artifact in &required {
        verify_package_v2_artifact(package_dir, artifact)?;
    }

    let sidecar_ids = resolved
        .sidecars
        .iter()
        .map(|sidecar| sidecar.artifact.id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let mut model_artifacts = required
        .iter()
        .filter(|artifact| !sidecar_ids.contains(artifact.id.as_str()))
        .collect::<Vec<_>>();
    model_artifacts.sort_by(|left, right| {
        let left_primary = left.id == manifest.source_model.metadata_artifact_id;
        let right_primary = right.id == manifest.source_model.metadata_artifact_id;
        right_primary
            .cmp(&left_primary)
            .then_with(|| left.id.cmp(&right.id))
    });
    let model_parts = model_artifacts
        .into_iter()
        .map(|artifact| package_dir.join(&artifact.path))
        .collect::<Vec<_>>();
    let projector = resolved
        .sidecars
        .iter()
        .find(|sidecar| sidecar.kind == skippy_package_format::SidecarKind::Mmproj)
        .map(|sidecar| package_dir.join(&sidecar.artifact.path));
    Ok((local_ref, model_parts, projector))
}

pub fn package_admission_descriptor(
    admission: &skippy_protocol::StageAdmissionDescriptor,
) -> PackageV2StageAdmissionDescriptor {
    skippy_package_format::stage_admission::StageAdmissionDescriptor {
        package_id: admission.package_id.clone(),
        resident_tensor_ids: admission.resident_tensor_ids.clone(),
        sidecars: admission
            .sidecars
            .iter()
            .map(|sidecar| skippy_package_format::Sidecar {
                kind: match sidecar.kind {
                    skippy_protocol::StageAdmissionSidecarKind::Mmproj => {
                        skippy_package_format::SidecarKind::Mmproj
                    }
                },
                artifact_id: sidecar.artifact_id.clone(),
                name: sidecar.name.clone(),
            })
            .collect(),
    }
}
pub fn resolve_admitted_stage_package(
    package_dir: &Path,
    expected_manifest_sha256: &str,
    admission: &skippy_protocol::StageAdmissionDescriptor,
) -> Result<crate::stage_load::ResolvedStagePackage> {
    let descriptor = package_admission_descriptor(admission);
    let (local_ref, model_part_paths, projector_path) =
        resolve_local_package_stage(package_dir, &descriptor)?;
    stage_package_from_verified_parts(
        local_ref,
        expected_manifest_sha256,
        model_part_paths,
        projector_path,
    )
}

/// Attach source facts to artifact paths already verified by `resolve_local_package_stage`.
/// Acquisition adapters use this after fetching and verifying required artifacts.
pub fn stage_package_from_verified_parts(
    local_ref: String,
    expected_manifest_sha256: &str,
    model_part_paths: Vec<PathBuf>,
    projector_path: Option<PathBuf>,
) -> Result<ResolvedStagePackage> {
    ensure_package_manifest_sha(&local_ref, expected_manifest_sha256)?;
    let manifest_path = Path::new(&local_ref).join("model-package.json");
    let manifest: skippy_package_format::PackageManifest = serde_json::from_slice(
        &fs::read(&manifest_path)
            .with_context(|| format!("read package-v2 manifest {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parse package-v2 manifest {}", manifest_path.display()))?;
    let source_model_bytes = manifest
        .source_model
        .files
        .iter()
        .try_fold(0_u64, |total, file| total.checked_add(file.byte_size))
        .context("package-v2 source byte count overflow")?;
    let source_model_path = model_part_paths
        .first()
        .context("package-v2 admission selected no model artifacts")?
        .to_string_lossy()
        .into_owned();
    Ok(ResolvedStagePackage {
        local_ref,
        source_model_path,
        source_model_sha256: manifest.source_model.sha256,
        source_model_bytes: Some(source_model_bytes),
        model_part_paths,
        projector_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    thread_local! {
        /// Files `file_sha256` hashed on this test's thread.
        pub(super) static HASHED: Cell<usize> = const { Cell::new(0) };
        /// How far this test's verification clock runs ahead of real time.
        pub(super) static CLOCK_AHEAD: Cell<Duration> = const { Cell::new(Duration::ZERO) };
    }

    /// Runs verifications on this thread with a clock far enough ahead that
    /// freshly written files count as settled.
    fn settled_clock() {
        CLOCK_AHEAD.with(|ahead| ahead.set(SETTLE_WINDOW * 300));
    }

    fn hashed() -> usize {
        HASHED.with(Cell::get)
    }

    fn artifact(path: &str, bytes: &[u8]) -> PackageV2Artifact {
        PackageV2Artifact {
            id: "layer-0".to_string(),
            path: path.to_string(),
            byte_size: bytes.len() as u64,
            sha256: hex::encode(Sha256::digest(bytes)),
        }
    }

    #[test]
    fn verifying_an_unchanged_artifact_again_skips_the_read() {
        settled_clock();
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("layer.gguf"), b"layer bytes").unwrap();
        let artifact = artifact("layer.gguf", b"layer bytes");
        let before = hashed();

        verify_package_v2_artifact(dir.path(), &artifact).unwrap();
        verify_package_v2_artifact(dir.path(), &artifact).unwrap();
        assert!(
            file_matches_sha256(
                &dir.path().join("layer.gguf"),
                &artifact.sha256.to_ascii_uppercase()
            )
            .unwrap()
        );
        assert_eq!(hashed(), before + 1, "an unchanged file is hashed once");
    }

    /// Hugging Face snapshots link to blobs: the blob is what is fingerprinted.
    #[cfg(unix)]
    #[test]
    fn a_linked_artifact_shares_the_verified_hash_of_its_target() {
        settled_clock();
        let dir = tempfile::tempdir().unwrap();
        let blob = dir.path().join("blob");
        fs::write(&blob, b"layer bytes").unwrap();
        std::os::unix::fs::symlink(&blob, dir.path().join("layer.gguf")).unwrap();
        let artifact = artifact("layer.gguf", b"layer bytes");
        let before = hashed();

        verify_package_v2_artifact(dir.path(), &artifact).unwrap();
        assert!(file_matches_sha256(&blob, &artifact.sha256).unwrap());
        assert_eq!(hashed(), before + 1);
    }

    #[test]
    fn a_size_change_is_hashed_again() {
        settled_clock();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("layer.gguf");
        fs::write(&path, b"layer bytes").unwrap();
        let old = artifact("layer.gguf", b"layer bytes").sha256;
        let new = artifact("layer.gguf", b"layer bytes, longer").sha256;
        let before = hashed();

        assert!(file_matches_sha256(&path, &old).unwrap());
        fs::write(&path, b"layer bytes, longer").unwrap();
        assert!(!file_matches_sha256(&path, &old).unwrap());
        assert!(file_matches_sha256(&path, &new).unwrap());
        assert_eq!(hashed(), before + 3);
    }

    #[test]
    fn a_same_size_rewrite_with_a_new_mtime_is_hashed_again_and_rejected() {
        settled_clock();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("layer.gguf");
        fs::write(&path, b"layer bytes").unwrap();
        let artifact = artifact("layer.gguf", b"layer bytes");
        verify_package_v2_artifact(dir.path(), &artifact).unwrap();
        let before = hashed();

        fs::write(&path, b"LAYER BYTES").unwrap();
        let later = SystemTime::now() + Duration::from_secs(60);
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(later)
            .unwrap();
        let error = verify_package_v2_artifact(dir.path(), &artifact).unwrap_err();
        assert!(error.to_string().contains("SHA-256 differs"), "{error:#}");
        assert_eq!(hashed(), before + 1);
    }

    /// Only the change time records a same-size rewrite whose mtime is put back.
    #[cfg(unix)]
    #[test]
    fn a_rewrite_with_its_mtime_restored_is_hashed_again_and_rejected() {
        settled_clock();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("layer.gguf");
        fs::write(&path, b"layer bytes").unwrap();
        let artifact = artifact("layer.gguf", b"layer bytes");
        verify_package_v2_artifact(dir.path(), &artifact).unwrap();
        let original = fs::metadata(&path).unwrap().modified().unwrap();
        let before = hashed();

        std::thread::sleep(Duration::from_millis(20));
        fs::write(&path, b"LAYER BYTES").unwrap();
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(original)
            .unwrap();
        assert!(verify_package_v2_artifact(dir.path(), &artifact).is_err());
        assert_eq!(hashed(), before + 1);
    }

    #[test]
    fn a_replaced_file_is_hashed_again() {
        settled_clock();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("layer.gguf");
        fs::write(&path, b"layer bytes").unwrap();
        let artifact = artifact("layer.gguf", b"layer bytes");
        verify_package_v2_artifact(dir.path(), &artifact).unwrap();
        let first =
            crate::source_registry::verified_path_fingerprint(std::slice::from_ref(&path)).unwrap();
        let before = hashed();

        // Same bytes and mtime in a new file renamed over the old one.
        let original = fs::metadata(&path).unwrap().modified().unwrap();
        let replacement = dir.path().join("replacement");
        fs::write(&replacement, b"layer bytes").unwrap();
        fs::File::options()
            .write(true)
            .open(&replacement)
            .unwrap()
            .set_modified(original)
            .unwrap();
        fs::rename(&replacement, &path).unwrap();
        let second =
            crate::source_registry::verified_path_fingerprint(std::slice::from_ref(&path)).unwrap();
        assert!(first != second, "a new file has a new identity");

        verify_package_v2_artifact(dir.path(), &artifact).unwrap();
        assert_eq!(hashed(), before + 1);
    }

    #[test]
    fn a_wrong_hash_is_still_rejected_after_a_cached_verification() {
        settled_clock();
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("layer.gguf"), b"layer bytes").unwrap();
        let good = artifact("layer.gguf", b"layer bytes");
        verify_package_v2_artifact(dir.path(), &good).unwrap();
        let wrong = PackageV2Artifact {
            sha256: artifact("layer.gguf", b"other bytes").sha256,
            ..good.clone()
        };
        let before = hashed();

        let error = verify_package_v2_artifact(dir.path(), &wrong).unwrap_err();
        assert!(error.to_string().contains("SHA-256 differs"), "{error:#}");
        assert!(verify_package_v2_artifact(dir.path(), &wrong).is_err());
        assert_eq!(hashed(), before + 2, "a mismatch is never cached");
        verify_package_v2_artifact(dir.path(), &good).unwrap();
        assert_eq!(hashed(), before + 2);
    }

    /// A file changed within the settle window is hashed on every check: an
    /// edit in the same timestamp tick could leave its fingerprint unchanged.
    #[test]
    fn a_recently_changed_artifact_is_hashed_on_every_check() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("layer.gguf"), b"layer bytes").unwrap();
        let artifact = artifact("layer.gguf", b"layer bytes");
        let before = hashed();

        verify_package_v2_artifact(dir.path(), &artifact).unwrap();
        verify_package_v2_artifact(dir.path(), &artifact).unwrap();
        assert_eq!(hashed(), before + 2, "a fresh file is not cached");
    }
}
