# Fixing Windows Linking Error (Exit Code 1120) for whisper.cpp Rust Wrapper

## The Problem

Getting this error when running tests (while build works fine):
```
error: linking with `link.exe` failed: exit code: 1120
```

Exit code 1120 means unresolved external symbols. The build succeeds but tests fail because the test binary can't find the whisper.cpp symbols. This is because on Windows, the build system needs to know where to find the compiled whisper.cpp library during both build AND runtime, and the test runner creates a separate binary that needs to link against the library.

## Solution 1: Update build.rs with Proper Linking Instructions (Most Likely Fix)

Your `build.rs` probably needs to output the correct linking instructions for Windows. Add these lines:

```rust
// In build.rs, after compiling the library
fn main() {
    // ... your existing build code ...
    
    // After cc::Build compiles the library
    let out_dir = env::var("OUT_DIR").unwrap();
    
    // Tell Rust where to find the library
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=whisper");
    
    // On Windows, we might need additional system libraries
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=dylib=ws2_32");
        println!("cargo:rustc-link-lib=dylib=bcrypt");
        println!("cargo:rustc-link-lib=dylib=advapi32");
        println!("cargo:rustc-link-lib=dylib=userenv");
    }
}
```

## Solution 2: Check Your cc::Build Configuration

Make sure your build.rs is setting up the Windows build correctly:

```rust
let mut build = cc::Build::new();

build
    .cpp(true)
    .std("c++11")
    .file("vendor/whisper.cpp/whisper.cpp")
    .file("vendor/whisper.cpp/ggml.c")
    .file("vendor/whisper.cpp/ggml-alloc.c")
    .file("vendor/whisper.cpp/ggml-backend.c")
    .file("vendor/whisper.cpp/ggml-quants.c");

// Windows-specific flags
if cfg!(target_os = "windows") {
    build
        .define("_CRT_SECURE_NO_WARNINGS", None)
        .define("NOMINMAX", None)
        .flag("/EHsc");  // Enable C++ exceptions
}

// Important: Set the output correctly
build.compile("whisper");  // This creates libwhisper.a or whisper.lib
```

## Solution 3: Diagnose Missing Symbols

To see what symbols are actually missing, run with verbose output:

```bash
cargo test --verbose 2>&1 | grep "unresolved external"
```

Common missing symbols and their fixes:

- **Missing C++ stdlib symbols**: Add `.cpp(true)` to your cc::Build
- **Missing Windows CRT symbols**: You might need to link against `msvcrt`
- **Missing whisper_* symbols**: The library isn't being built or linked correctly
- **Missing ggml_* symbols**: Make sure you're compiling all necessary ggml files

## Solution 4: Use Static CRT for Windows

Add this to your `.cargo/config.toml` in the project root:

```toml
[target.x86_64-pc-windows-msvc]
rustflags = ["-C", "target-feature=+crt-static"]
```

This statically links the C runtime, avoiding DLL issues.

## Solution 5: Complete Working build.rs for Windows

Here's a complete working build.rs that handles Windows correctly:

```rust
use std::env;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    // Build whisper.cpp
    let mut build = cc::Build::new();
    
    build
        .cpp(true)
        .std("c++11")
        .include("vendor/whisper.cpp")
        .file("vendor/whisper.cpp/whisper.cpp")
        .file("vendor/whisper.cpp/ggml.c")
        .file("vendor/whisper.cpp/ggml-alloc.c")
        .file("vendor/whisper.cpp/ggml-backend.c")
        .file("vendor/whisper.cpp/ggml-quants.c");
    
    // Windows specific settings
    #[cfg(target_os = "windows")]
    {
        build
            .define("_CRT_SECURE_NO_WARNINGS", None)
            .define("WIN32_LEAN_AND_MEAN", None)
            .define("NOMINMAX", None)
            .flag("/EHsc")
            .flag("/MT");  // Static runtime
    }
    
    // macOS specific settings
    #[cfg(target_os = "macos")]
    {
        build.flag("-std=c++14");
    }
    
    // Linux specific settings
    #[cfg(target_os = "linux")]
    {
        build.flag("-std=c++11");
    }
    
    build.compile("whisper");
    
    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header("vendor/whisper.cpp/whisper.h")
        .clang_arg("-x").clang_arg("c++")
        .clang_arg("-std=c++11")
        .allowlist_function("whisper_.*")
        .allowlist_type("whisper_.*")
        .opaque_type("std::.*")
        .generate()
        .expect("Unable to generate bindings");
    
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings");
    
    // Linking instructions
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=whisper");
    
    #[cfg(target_os = "windows")]
    {
        // Windows system libraries
        println!("cargo:rustc-link-lib=ws2_32");
        println!("cargo:rustc-link-lib=bcrypt");
        println!("cargo:rustc-link-lib=advapi32");
        println!("cargo:rustc-link-lib=userenv");
    }
    
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=c++");
    }
    
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
```

## Quick Debug Steps

1. **Clean and rebuild**:
```bash
cargo clean
cargo build --verbose
cargo test --verbose
```

2. **Check that the library was actually created**:
```bash
# Windows
dir target\debug\build\*\out\*.lib

# Linux/macOS
ls target/debug/build/*/out/*.a
```

3. **Try building whisper.cpp separately** to ensure it compiles on your system:
```bash
cd vendor/whisper.cpp
mkdir build && cd build
cmake ..
cmake --build .
```

## Common Windows-Specific Issues

### Issue: Missing MSVC Build Tools
**Symptom**: Can't find cl.exe or link.exe  
**Fix**: Install Visual Studio Build Tools 2022 with C++ workload

### Issue: Wrong MSVC Toolchain
**Symptom**: Incompatible .lib format  
**Fix**: Ensure you're using the x64 toolchain:
```bash
rustup default stable-x86_64-pc-windows-msvc
```

### Issue: Missing Windows SDK
**Symptom**: Can't find Windows headers  
**Fix**: Install Windows SDK through Visual Studio Installer

### Issue: Conflicting Runtime Libraries
**Symptom**: Multiple definitions of CRT functions  
**Fix**: Use consistent `/MT` (static) or `/MD` (dynamic) flags

## Alternative: Use CMake Instead of cc

If cc::Build continues to cause issues, you can use cmake crate instead:

```rust
// In Cargo.toml build-dependencies
[build-dependencies]
cmake = "0.1"

// In build.rs
use cmake::Config;

fn main() {
    let dst = Config::new("vendor/whisper.cpp")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("WHISPER_BUILD_TESTS", "OFF")
        .define("WHISPER_BUILD_EXAMPLES", "OFF")
        .build();
    
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=whisper");
    
    // Windows libraries
    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-link-lib=ws2_32");
        println!("cargo:rustc-link-lib=bcrypt");
    }
}
```

## Testing the Fix

After applying these solutions, test with a minimal example:

```rust
// In tests/linking_test.rs
#[test]
fn test_whisper_version() {
    unsafe {
        let version = whisper_sys::whisper_full_default_params(
            whisper_sys::whisper_sampling_strategy_WHISPER_SAMPLING_GREEDY
        );
        // If this doesn't crash, linking works!
        assert_eq!(version.n_threads, 4); // Or whatever the default is
    }
}
```

## Priority Order for Fixes

1. **First**: Try Solution 1 (add linking instructions to build.rs)
2. **Second**: Ensure all ggml files are being compiled (Solution 2)
3. **Third**: Add static CRT linking (Solution 4)
4. **Fourth**: Use the complete build.rs from Solution 5
5. **Last Resort**: Switch to CMake-based build

The most common cause is missing linking instructions in build.rs. The Windows linker needs explicit instructions about where to find the compiled library and which system libraries to link against.