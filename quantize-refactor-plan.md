# Quantization Refactoring Plan

## Overview
Refactor the quantization functionality from `whisper-cpp-rs` into a separate `whisper-quantize` crate to better align with whisper.cpp's architecture and provide cleaner separation of concerns.

## Motivation
- **Architectural Alignment**: whisper.cpp keeps quantization as a separate tool in `examples/quantize/`, not part of the main API
- **Separation of Concerns**: Quantization is a build-time/deployment tool, not a runtime feature
- **Reduced Dependencies**: Users who only need inference shouldn't pay for quantization code (~700 lines)
- **Optional Feature**: Makes quantization opt-in rather than always included

## Current Structure
```
whisper-cpp-rs/
├── src/
│   ├── quantize.rs (440 lines)    # To be moved
│   └── lib.rs (exports quantization)
whisper-sys/
├── src/
│   ├── quantize_wrapper.cpp (282 lines)  # Keep, make optional
│   └── lib.rs (FFI bindings)
```

## Proposed Structure
```
workspace-root/
├── whisper-cpp-rs/       # Main transcription API (no quantization)
├── whisper-quantize/     # New: Quantization utilities
├── whisper-sys/          # Shared FFI bindings
└── xtask/                # Build tools
```

## Implementation Steps

### Step 1: Create `whisper-quantize` crate

**Directory structure:**
```
whisper-quantize/
├── Cargo.toml
├── src/
│   └── lib.rs           # Move from whisper-cpp-rs/src/quantize.rs
├── tests/
│   └── quantization.rs  # Move from whisper-cpp-rs/tests/quantization.rs
└── examples/
    └── quantize_model.rs # New example showing usage
```

**`whisper-quantize/Cargo.toml`:**
```toml
[package]
name = "whisper-quantize"
version = "0.1.0"
edition = "2021"
description = "Model quantization utilities for whisper-cpp-rs"
license = "MIT"
repository = "https://github.com/yourusername/whisper-cpp-rs"
keywords = ["whisper", "quantization", "compression", "ml", "optimization"]
categories = ["science", "compression"]

[dependencies]
whisper-sys = { path = "../whisper-sys", features = ["quantization"] }
thiserror = "1.0"

[dev-dependencies]
whisper-cpp-rs = { path = "../whisper-cpp-rs" }
```

### Step 2: Update `whisper-sys` for optional quantization

**Changes to `whisper-sys/Cargo.toml`:**
```toml
[features]
default = []
quantization = []  # New feature flag
```

**Changes to `whisper-sys/build.rs`:**
```rust
fn build_whisper_cpp(target_os: &str, target_arch: &str) {
    // ... existing setup ...

    // Core source files
    build.file("../vendor/whisper.cpp/src/whisper.cpp")
        // ... existing GGML files ...

    // Conditionally add quantization support
    #[cfg(feature = "quantization")]
    {
        build.file("../vendor/whisper.cpp/examples/common.cpp")
            .file("../vendor/whisper.cpp/examples/common-ggml.cpp")
            .file("src/quantize_wrapper.cpp");
    }

    // ... rest of build config ...
}
```

**Changes to `whisper-sys/src/lib.rs`:**
```rust
// ... existing exports ...

#[cfg(feature = "quantization")]
pub mod quantization {
    // Move quantization-specific constants and FFI here
    pub const WHISPER_QUANTIZE_OK: i32 = 0;
    // ... etc

    extern "C" {
        pub fn whisper_model_quantize(...);
        pub fn whisper_model_get_ftype(...);
    }
}
```

### Step 3: Clean up `whisper-cpp-rs`

**Remove from `whisper-cpp-rs/src/lib.rs`:**
- `mod quantize;`
- `pub use quantize::{ModelQuantizer, QuantizationType};`

**Remove from `whisper-cpp-rs/src/context.rs`:**
- `use crate::quantize::{ModelQuantizer, QuantizationType};`
- `WhisperContext::quantize_model()` method

**Move test file:**
- `whisper-cpp-rs/tests/quantization.rs` → `whisper-quantize/tests/quantization.rs`

### Step 4: Update workspace configuration

**Update root `Cargo.toml`:**
```toml
[workspace]
members = [
    "whisper-sys",
    "whisper-cpp-rs",
    "whisper-quantize",  # New member
    "xtask"
]
```

### Step 5: Create migration documentation

**Add to README.md:**
```markdown
## Crate Structure

- **`whisper-cpp-rs`** - Main transcription API
- **`whisper-quantize`** - Model quantization utilities (optional)
- **`whisper-sys`** - Low-level FFI bindings

### Using Quantization

Model quantization is available as a separate crate:

```toml
[dependencies]
whisper-cpp-rs = "0.1"
whisper-quantize = "0.1"  # Only if you need quantization
```

```rust
use whisper_quantize::{ModelQuantizer, QuantizationType};

ModelQuantizer::quantize_model_file(
    "models/ggml-base.bin",
    "models/ggml-base-q5_0.bin",
    QuantizationType::Q5_0
)?;
```
```

### Step 6: Update examples

Create `whisper-quantize/examples/quantize_model.rs`:
```rust
use whisper_quantize::{ModelQuantizer, QuantizationType};
use std::env;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <input_model> <output_model> <quantization_type>", args[0]);
        eprintln!("Quantization types: q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k, q4_k, q5_k, q6_k");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let qtype = QuantizationType::from_str(&args[3])
        .ok_or("Invalid quantization type")?;

    println!("Quantizing {} to {} using {}", input_path, output_path, qtype);

    ModelQuantizer::quantize_model_file_with_progress(
        input_path,
        output_path,
        qtype,
        |progress| {
            print!("\rProgress: {:.1}%", progress * 100.0);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
    )?;

    println!("\nQuantization complete!");
    Ok(())
}
```

## Benefits

1. **Cleaner API Surface**: Main crate focuses solely on transcription
2. **Optional Dependency**: Users choose whether to include quantization
3. **Smaller Binary Size**: ~700 lines of code only included when needed
4. **Better Alignment**: Matches whisper.cpp's architectural decisions
5. **Independent Versioning**: Quantization utilities can evolve separately

## Migration Guide for Users

### Before (integrated):
```rust
use whisper_cpp_rs::{WhisperContext, QuantizationType};

WhisperContext::quantize_model(input, output, QuantizationType::Q5_0)?;
```

### After (separate crate):
```rust
use whisper_quantize::{ModelQuantizer, QuantizationType};

ModelQuantizer::quantize_model_file(input, output, QuantizationType::Q5_0)?;
```

## Testing Plan

1. Ensure `whisper-sys` builds with and without `quantization` feature
2. Verify `whisper-cpp-rs` works without quantization code
3. Test `whisper-quantize` crate independently
4. Verify prebuild system handles optional quantization correctly
5. Update CI to test both configurations

## Rollout Plan

1. **Phase 1**: Create `whisper-quantize` crate with existing code
2. **Phase 2**: Make quantization optional in `whisper-sys`
3. **Phase 3**: Remove quantization from `whisper-cpp-rs`
4. **Phase 4**: Update documentation and examples
5. **Phase 5**: Test thoroughly before release

## Open Questions

1. Should we provide a compatibility shim in `whisper-cpp-rs` that redirects to `whisper-quantize`?
2. Should the prebuild system always include quantization support or make it configurable?
3. Do we want to add a CLI binary to `whisper-quantize` for command-line usage?

## Timeline

Estimated effort: 2-3 hours
- Step 1-2: 1 hour (create new crate, update whisper-sys)
- Step 3-4: 30 minutes (clean up main crate)
- Step 5-6: 30 minutes (documentation and examples)
- Testing: 1 hour

## Decision Record

**Date**: 2024-01-22
**Decision**: Separate quantization into its own crate
**Rationale**: Better architectural alignment with whisper.cpp, cleaner separation of concerns, optional dependency