# Publishing Guide

Guide for publishing whisper-cpp-plus crates to crates.io.

## Pre-publish Checklist

### 1. Version Bump

Update version in all locations:

```bash
# Workspace version (root Cargo.toml)
# whisper-cpp-plus-sys dependency version (whisper-cpp-plus/Cargo.toml)
# README.md examples (root + whisper-cpp-plus/)
# Doc comments (whisper-cpp-plus/src/quantize.rs)
```

### 2. Test docs.rs Build Locally

docs.rs runs in a **network-isolated container** - it cannot download dependencies at build time. Our `build.rs` detects `DOCS_RS=1` and generates stub bindings instead of compiling whisper.cpp.

**Test the stub bindings work:**

```bash
# Clean and rebuild with DOCS_RS simulation
export DOCS_RS=1
cargo clean -p whisper-cpp-plus-sys
cargo check -p whisper-cpp-plus

# Test docs generation
cargo doc -p whisper-cpp-plus --no-deps
```

If this fails, the stub bindings in `whisper-cpp-plus-sys/build.rs` (`generate_stub_bindings()`) need updating to include missing FFI symbols.

### 3. Run Tests

```bash
cargo test -p whisper-cpp-plus
cargo test -p whisper-cpp-plus --features async
```

### 4. Verify Package Contents

```bash
cargo package -p whisper-cpp-plus-sys --list
cargo package -p whisper-cpp-plus --list
```

## Publishing

**Order matters** - sys crate must be published first:

```bash
# 1. Publish sys crate
cargo publish -p whisper-cpp-plus-sys

# 2. Publish main crate
cargo publish -p whisper-cpp-plus
```

## Git Tags & GitHub Releases

After publishing, create matching git tags:

```bash
# Tag current commit
git tag -a v0.1.X -m "v0.1.X: Brief description"
git push origin v0.1.X

# Create GitHub release
gh release create v0.1.X --title "v0.1.X" --notes "Release notes here"
```

## Verifying docs.rs Build

After publishing, monitor the docs.rs build:

1. Check build queue: https://docs.rs/releases/queue
2. View build status: https://docs.rs/crate/whisper-cpp-plus/VERSION/builds
3. If build fails, check logs and fix stub bindings

### Common docs.rs Failures

| Error | Cause | Fix |
|-------|-------|-----|
| DNS resolution failed | Network access attempted | Ensure `DOCS_RS` check in build.rs |
| Cannot find function X | Missing stub binding | Add function to `generate_stub_bindings()` |
| Type mismatch | Stub signature wrong | Match stub to actual usage in high-level crate |
| Inner attribute not permitted | `#![allow(...)]` in included file | Remove inner attrs from stub bindings |

## Stub Bindings Maintenance

When adding new FFI functions to the high-level crate, also add stubs:

1. Add function to `generate_stub_bindings()` in `whisper-cpp-plus-sys/build.rs`
2. Match the signature to how the high-level code calls it
3. Test with `DOCS_RS=1 cargo check -p whisper-cpp-plus`

## Yanking Bad Releases

If a release has critical issues:

```bash
cargo yank --version 0.1.X whisper-cpp-plus
cargo yank --version 0.1.X whisper-cpp-plus-sys
```

Note: Yanked versions can still be used by existing Cargo.lock files but won't be selected for new projects.
