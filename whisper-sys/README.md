# whisper-sys ⚙️

Low-level FFI bindings to [whisper.cpp](https://github.com/ggerganov/whisper.cpp) for Rust.

[![Crates.io](https://img.shields.io/crates/v/whisper-sys.svg)](https://crates.io/crates/whisper-sys)
[![Documentation](https://docs.rs/whisper-sys/badge.svg)](https://docs.rs/whisper-sys)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE)

## Overview

`whisper-sys` provides raw, unsafe Rust bindings to whisper.cpp's C API. This crate is the foundation for [whisper-cpp-rs](https://github.com/yourusername/whisper-cpp-rs), but can be used directly when you need:

- Maximum control over whisper.cpp functionality
- Integration with existing C/C++ code
- Custom memory management
- Access to experimental or bleeding-edge features
- Building your own safe wrapper

⚠️ **Warning:** This crate provides raw FFI bindings. All functions are `unsafe` and require careful handling of pointers, lifetimes, and memory management.

## When to Use

| Use Case | Recommended Crate |
|----------|------------------|
| General transcription | `whisper-cpp-rs` (safe wrapper) |
| Simple API with safety | `whisper-cpp-rs` |
| Maximum performance control | `whisper-sys` (this crate) |
| Custom memory management | `whisper-sys` |
| C/C++ interop | `whisper-sys` |
| Experimental features | `whisper-sys` |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
whisper-sys = "0.1.0"
```

### Features

```toml
[dependencies]
whisper-sys = { version = "0.1.0", features = ["cuda"] }
```

Available features:
- `quantization` - Enable model quantization support
- `cuda` - NVIDIA GPU acceleration (requires CUDA toolkit)
- `metal` - Apple Metal acceleration (macOS only)
- `openblas` - OpenBLAS acceleration

## API Overview

### Core Types

The crate exposes whisper.cpp's core types through bindgen:

```rust
// Opaque pointer types (must be managed manually)
pub struct whisper_context;
pub struct whisper_state;

// Configuration structures
#[repr(C)]
pub struct whisper_context_params {
    pub use_gpu: bool,
    pub flash_attn: bool,
    pub gpu_device: c_int,
    pub dtw_mem_size: usize,
    // ...
}

#[repr(C)]
pub struct whisper_full_params {
    pub strategy: c_int,
    pub n_threads: c_int,
    pub n_max_text_ctx: c_int,
    pub offset_ms: c_int,
    pub duration_ms: c_int,
    // ...
}
```

### Function Categories

| Category | Functions | Purpose |
|----------|-----------|---------|
| **Initialization** | `whisper_init_*`, `whisper_free*` | Model loading and cleanup |
| **Transcription** | `whisper_full*`, `whisper_decode*` | Audio processing |
| **Results** | `whisper_full_get_*` | Access transcription results |
| **Model Info** | `whisper_model_*`, `whisper_n_*` | Model introspection |
| **Tokenization** | `whisper_tokenize*`, `whisper_token_*` | Token manipulation |
| **Language** | `whisper_lang_*` | Language detection/info |

## Basic Usage

### Loading a Model

```rust
use whisper_sys::*;
use std::ffi::CString;
use std::ptr;

unsafe {
    // Initialize with default parameters
    let mut params = whisper_context_default_params();
    params.use_gpu = true;  // Enable GPU if available

    // Load model from file
    let model_path = CString::new("models/ggml-base.bin").unwrap();
    let ctx = whisper_init_from_file_with_params(
        model_path.as_ptr(),
        params
    );

    if ctx.is_null() {
        panic!("Failed to load model");
    }

    // Use the context...

    // Clean up
    whisper_free(ctx);
}
```

### Transcribing Audio

```rust
use whisper_sys::*;
use std::slice;

unsafe fn transcribe(ctx: *mut whisper_context, audio: &[f32]) -> Vec<String> {
    // Create state for this transcription
    let state = whisper_init_state(ctx);
    if state.is_null() {
        panic!("Failed to create state");
    }

    // Configure parameters
    let mut params = whisper_full_default_params(
        WHISPER_SAMPLING_GREEDY
    );
    params.n_threads = 4;
    params.language = CString::new("en").unwrap().as_ptr();

    // Run transcription
    let result = whisper_full(
        ctx,
        state,
        params,
        audio.as_ptr(),
        audio.len() as c_int
    );

    if result != 0 {
        whisper_free_state(state);
        panic!("Transcription failed");
    }

    // Extract segments
    let n_segments = whisper_full_n_segments_from_state(state);
    let mut segments = Vec::new();

    for i in 0..n_segments {
        let text_ptr = whisper_full_get_segment_text_from_state(state, i);
        if !text_ptr.is_null() {
            let text = CStr::from_ptr(text_ptr)
                .to_string_lossy()
                .into_owned();
            segments.push(text);
        }
    }

    // Clean up
    whisper_free_state(state);

    segments
}
```

### Advanced: Custom Callbacks

```rust
use whisper_sys::*;
use std::os::raw::{c_void, c_int};

// Progress callback
unsafe extern "C" fn progress_callback(
    _ctx: *mut whisper_context,
    _state: *mut whisper_state,
    progress: c_int,
    user_data: *mut c_void
) {
    let progress_percent = progress as f32;
    println!("Progress: {:.1}%", progress_percent);
}

// Segment callback for streaming
unsafe extern "C" fn new_segment_callback(
    _ctx: *mut whisper_context,
    state: *mut whisper_state,
    n_new: c_int,
    user_data: *mut c_void
) {
    let n_segments = whisper_full_n_segments_from_state(state);

    // Process only new segments
    for i in (n_segments - n_new)..n_segments {
        let text_ptr = whisper_full_get_segment_text_from_state(state, i);
        if !text_ptr.is_null() {
            let text = CStr::from_ptr(text_ptr).to_string_lossy();
            println!("New segment: {}", text);
        }
    }
}

// Use callbacks
unsafe {
    let mut params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.progress_callback = Some(progress_callback);
    params.new_segment_callback = Some(new_segment_callback);
    params.new_segment_callback_user_data = ptr::null_mut();

    // Run transcription with callbacks...
}
```

### Using Quantization Feature

```rust
#[cfg(feature = "quantization")]
use whisper_sys::{whisper_model_quantize, GGML_FTYPE_MOSTLY_Q5_0};

#[cfg(feature = "quantization")]
unsafe fn quantize_model(input: &str, output: &str) -> Result<(), String> {
    let input_path = CString::new(input).unwrap();
    let output_path = CString::new(output).unwrap();

    let result = whisper_model_quantize(
        input_path.as_ptr(),
        output_path.as_ptr(),
        GGML_FTYPE_MOSTLY_Q5_0,
        None  // No progress callback
    );

    if result == WHISPER_QUANTIZE_OK {
        Ok(())
    } else {
        Err(format!("Quantization failed with code: {}", result))
    }
}
```

## Memory Management

### Rules for Safe Usage

1. **Context Lifetime**: `whisper_context` must outlive all `whisper_state` instances
2. **State Isolation**: Each thread needs its own `whisper_state`
3. **String Lifetime**: Strings returned by whisper.cpp are owned by the context/state
4. **Cleanup Order**: Free states before context

```rust
// Correct cleanup order
unsafe {
    let ctx = whisper_init_from_file(path);
    let state1 = whisper_init_state(ctx);
    let state2 = whisper_init_state(ctx);

    // Use states...

    whisper_free_state(state2);  // Free states first
    whisper_free_state(state1);
    whisper_free(ctx);           // Then free context
}
```

### Thread Safety

- `whisper_context` is read-only after creation (can be shared)
- `whisper_state` is NOT thread-safe (one per thread)
- Use separate states for concurrent transcription

```rust
use std::sync::Arc;
use std::thread;

unsafe {
    let ctx = whisper_init_from_file(path);
    let ctx_ptr = Arc::new(ctx);

    let handles: Vec<_> = (0..4).map(|i| {
        let ctx = Arc::clone(&ctx_ptr);
        thread::spawn(move || {
            let state = whisper_init_state(*ctx);
            // Do transcription...
            whisper_free_state(state);
        })
    }).collect();

    for handle in handles {
        handle.join().unwrap();
    }

    whisper_free(*ctx_ptr);
}
```

## Build Configuration

### Environment Variables

Control the build process with environment variables:

```bash
# Use prebuilt library instead of building from source
export WHISPER_PREBUILT_PATH=/path/to/prebuilt

# Custom compiler flags
export WHISPER_CFLAGS="-O3 -march=native"

# Disable specific features
export WHISPER_NO_AVX2=1
```

### Platform-Specific Notes

#### Windows (MSVC)
- Requires Visual Studio 2019 or later
- Automatically links required Windows libraries
- Builds with `/MT` flag for static runtime

#### Linux
- Requires GCC 9+ or Clang 11+
- OpenBLAS acceleration available via feature flag
- May need to install development packages:
  ```bash
  sudo apt-get install build-essential
  ```

#### macOS
- Accelerate framework linked automatically
- Metal support available via feature flag
- Universal binaries supported for Intel/Apple Silicon

### Cross-Compilation

```toml
# In .cargo/config.toml
[target.aarch64-unknown-linux-gnu]
linker = "aarch64-linux-gnu-gcc"

[target.armv7-unknown-linux-gnueabihf]
linker = "arm-linux-gnueabihf-gcc"
```

## Error Codes

Custom error codes for common failures:

```rust
use whisper_sys::*;

match result {
    WHISPER_ERR_INVALID_MODEL => "Invalid or corrupted model",
    WHISPER_ERR_NOT_ENOUGH_MEMORY => "Insufficient memory",
    WHISPER_ERR_FAILED_TO_PROCESS => "Processing failed",
    WHISPER_ERR_INVALID_CONTEXT => "Invalid context",
    _ if result < 0 => "Unknown error",
    _ => "Success"
}
```

## Debugging

Enable debug output:

```rust
unsafe {
    // Enable verbose logging (if compiled with logging support)
    whisper_log_set(Some(log_callback), ptr::null_mut());
}

unsafe extern "C" fn log_callback(
    level: c_int,
    text: *const c_char,
    user_data: *mut c_void
) {
    let message = CStr::from_ptr(text).to_string_lossy();
    eprintln!("[whisper.cpp {}]: {}", level, message);
}
```

## Performance Tips

1. **Batch Processing**: Process multiple files with same context
2. **Thread Count**: Set `n_threads` to physical core count
3. **GPU Acceleration**: Enable CUDA/Metal features when available
4. **Memory Mapping**: Use memory-mapped models for large files
5. **Quantization**: Use quantized models for faster inference

## Common Pitfalls

### ❌ Don't Do This

```rust
// WRONG: Using freed memory
unsafe {
    let text = {
        let state = whisper_init_state(ctx);
        let ptr = whisper_full_get_segment_text_from_state(state, 0);
        whisper_free_state(state);
        CStr::from_ptr(ptr).to_string_lossy()  // Use after free!
    };
}
```

### ✅ Do This Instead

```rust
// CORRECT: Copy before freeing
unsafe {
    let text = {
        let state = whisper_init_state(ctx);
        let ptr = whisper_full_get_segment_text_from_state(state, 0);
        let text = CStr::from_ptr(ptr).to_string_lossy().into_owned();
        whisper_free_state(state);
        text  // Owned copy is safe
    };
}
```

## Examples

See the [examples](./examples) directory for more complete examples:
- `basic.rs` - Simple transcription
- `streaming.rs` - Real-time streaming with callbacks
- `parallel.rs` - Multi-threaded processing
- `custom_model.rs` - Loading models with custom parameters

## Contributing

When contributing to whisper-sys:
1. Ensure bindgen compatibility
2. Test on multiple platforms
3. Document any new manual bindings
4. Update feature flags if needed

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Safety Notice

This crate provides **unsafe** FFI bindings. Users are responsible for:
- Proper memory management
- Thread safety
- Null pointer checks
- Lifetime management

For a safe, high-level API, use [whisper-cpp-rs](https://github.com/yourusername/whisper-cpp-rs) instead.