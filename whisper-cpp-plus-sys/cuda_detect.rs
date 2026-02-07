#![allow(dead_code)]
//! CUDA toolkit detection for build scripts.
//!
//! Finds CUDA toolkit library directories via environment variables
//! and platform-specific standard locations.
//!
//! Adapted from cudarc (https://github.com/coreylowman/cudarc).

use std::env;
use std::path::PathBuf;

/// Environment variables that may point to a CUDA toolkit installation.
pub const CUDA_PATH_ENV_VARS: [&str; 4] = [
    "CUDA_PATH",
    "CUDA_HOME",
    "CUDA_ROOT",
    "CUDA_TOOLKIT_ROOT_DIR",
];

/// Standard CUDA installation locations to probe when no env var is set.
const STANDARD_ROOTS: &[&str] = &[
    "/usr",
    "/usr/local/cuda",
    "/opt/cuda",
    "/usr/lib/cuda",
    "C:/Program Files/NVIDIA GPU Computing Toolkit",
    "C:/Program Files/NVIDIA",
    "C:/CUDA",
];

/// Library subdirectory patterns found in CUDA toolkit installations.
const LIB_SUBDIRS: &[&str] = &[
    "lib",
    "lib/x64",
    "lib/Win32",
    "lib/x86_64",
    "lib/x86_64-linux-gnu",
    "lib/stubs",
    "lib64",
    "lib64/stubs",
    "targets/x86_64-linux/lib",
    "targets/x86_64-linux/lib/stubs",
];

/// Find CUDA toolkit root directories from environment variables.
/// Returns roots found via env vars, or empty vec if none set.
pub fn cuda_roots_from_env() -> Vec<PathBuf> {
    CUDA_PATH_ENV_VARS
        .iter()
        .filter_map(|var| env::var(var).ok())
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .collect()
}

/// Find CUDA toolkit library directories.
///
/// Strategy:
/// 1. Check env vars (CUDA_PATH, CUDA_HOME, CUDA_ROOT, CUDA_TOOLKIT_ROOT_DIR)
/// 2. Fall back to platform-specific standard locations
/// 3. For each root, probe known lib subdirectory layouts
///
/// Returns all existing lib directories found, or empty vec if nothing found.
pub fn cuda_lib_search_paths() -> Vec<PathBuf> {
    let env_roots = cuda_roots_from_env();

    let roots: Vec<PathBuf> = if env_roots.is_empty() {
        STANDARD_ROOTS
            .iter()
            .map(PathBuf::from)
            .filter(|p| p.exists())
            .collect()
    } else {
        env_roots
    };

    let mut candidates = Vec::new();
    for root in &roots {
        for subdir in LIB_SUBDIRS {
            let path = root.join(subdir);
            if path.is_dir() {
                candidates.push(path);
            }
        }
    }

    candidates
}
