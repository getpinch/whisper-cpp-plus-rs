# whisper-quantize 🗜️

Model quantization utilities for [whisper-cpp-rs](https://github.com/yourusername/whisper-cpp-rs), providing tools to compress Whisper models for efficient deployment.

[![Crates.io](https://img.shields.io/crates/v/whisper-quantize.svg)](https://crates.io/crates/whisper-quantize)
[![Documentation](https://docs.rs/whisper-quantize/badge.svg)](https://docs.rs/whisper-quantize)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE)

## Overview

`whisper-quantize` reduces Whisper model sizes by 50-75% through quantization, enabling deployment on resource-constrained devices while maintaining acceptable accuracy. This crate provides the same quantization capabilities as whisper.cpp's quantization tools.

## Quick Start

```rust
use whisper_quantize::{ModelQuantizer, QuantizationType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Quantize a model to 5-bit precision
    ModelQuantizer::quantize_model_file(
        "models/ggml-base.bin",
        "models/ggml-base-q5_0.bin",
        QuantizationType::Q5_0
    )?;

    println!("Model quantized successfully!");
    Ok(())
}
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
whisper-quantize = "0.1.0"
```

## Quantization Types

### Standard Quantization (Q-series)

| Type | Bits | Size Reduction | Speed | Quality | Use Case |
|------|------|---------------|-------|---------|----------|
| `Q4_0` | 4-bit | ~75% | Fastest | Good | Mobile/embedded devices |
| `Q4_1` | 4-bit | ~72% | Fast | Good+ | Balanced performance |
| `Q5_0` | 5-bit | ~70% | Fast | Very Good | **Recommended default** |
| `Q5_1` | 5-bit | ~68% | Fast | Very Good+ | Higher quality, slightly larger |
| `Q8_0` | 8-bit | ~50% | Medium | Excellent | High quality, moderate size |

### K-Quantization (K-series)

K-quantization uses more sophisticated techniques for better quality at similar compression ratios:

| Type | Bits | Size Reduction | Quality | Use Case |
|------|------|---------------|---------|----------|
| `Q2_K` | 2-bit | ~85% | Fair | Extreme compression |
| `Q3_K` | 3-bit | ~80% | Good | High compression |
| `Q4_K` | 4-bit | ~75% | Very Good | **Best for most K-series uses** |
| `Q5_K` | 5-bit | ~70% | Excellent | High quality K-quantization |
| `Q6_K` | 6-bit | ~65% | Excellent+ | Maximum K-series quality |

## API Usage

### Basic Quantization

```rust
use whisper_quantize::{ModelQuantizer, QuantizationType};

// Simple quantization
ModelQuantizer::quantize_model_file(
    "input.bin",
    "output.bin",
    QuantizationType::Q5_0
)?;
```

### With Progress Tracking

```rust
use whisper_quantize::{ModelQuantizer, QuantizationType};

ModelQuantizer::quantize_model_file_with_progress(
    "input.bin",
    "output.bin",
    QuantizationType::Q5_0,
    |progress| {
        println!("Progress: {:.1}%", progress * 100.0);
    }
)?;
```

### Detecting Model Quantization

```rust
use whisper_quantize::ModelQuantizer;

// Check if a model is already quantized
match ModelQuantizer::get_model_ftype("model.bin")? {
    Some(qtype) => println!("Model is quantized as: {}", qtype),
    None => println!("Model is not quantized (F32/F16)"),
}
```

### Estimating Output Size

```rust
use whisper_quantize::{ModelQuantizer, QuantizationType};

// Get estimated size before quantization
let input_size = ModelQuantizer::estimate_model_size("input.bin")?;
let output_size = (input_size as f64 * QuantizationType::Q5_0.size_factor()) as u64;

println!("Estimated output size: {:.2} MB", output_size as f64 / 1_048_576.0);
```

## Command-Line Tool

The crate includes a command-line tool for quantizing models:

```bash
# Install the tool
cargo install whisper-quantize

# Basic usage
whisper-quantize input_model.bin output_model.bin q5_0

# Run from source
cargo run --example quantize_model -- input.bin output.bin q5_0
```

### Example Output

```
Quantizing model:
  Input:  models/ggml-base.bin (147.95 MB)
  Output: models/ggml-base-q5_0.bin
  Type:   Q5_0 (5-bit quantization (method 0))

Progress: 100.0%

Quantization complete!
  Output size: 44.39 MB
  Size reduction: 70.0%
  Compression ratio: 3.33x
```

## Model Size Comparison

Example sizes for the base model (~148 MB unquantized):

| Quantization | Size (MB) | Reduction |
|--------------|-----------|-----------|
| Original (F16) | 147.95 | 0% |
| Q8_0 | 78.73 | 46.8% |
| Q5_1 | 50.68 | 65.7% |
| Q5_0 | 44.39 | 70.0% |
| Q4_1 | 41.24 | 72.1% |
| Q4_0 | 36.97 | 75.0% |
| Q4_K | 39.59 | 73.2% |
| Q3_K | 32.09 | 78.3% |
| Q2_K | 24.39 | 83.5% |

## Best Practices

### Choosing a Quantization Type

1. **For general use**: Start with `Q5_0` - excellent balance of size, speed, and quality
2. **For mobile/embedded**: Use `Q4_0` or `Q4_K` - maximum compression with good quality
3. **For quality-critical**: Use `Q8_0` - minimal quality loss
4. **For experimentation**: Try K-series (`Q4_K`, `Q5_K`) - often better quality at same size

### Performance Tips

- Quantize models once and distribute the quantized versions
- Test accuracy with your specific use case - impact varies by model and domain
- Smaller models (tiny, base) tolerate quantization better than larger ones
- Consider keeping the original model for comparison/fallback

### Quality Considerations

Quantization impact on accuracy (typical):
- Q8_0: < 0.1% accuracy loss
- Q5_0/Q5_1: ~0.5-1% accuracy loss
- Q4_0/Q4_1: ~1-2% accuracy loss
- Q3_K: ~2-3% accuracy loss
- Q2_K: ~3-5% accuracy loss

## Error Handling

```rust
use whisper_quantize::{ModelQuantizer, QuantizationType, QuantizeError};

match ModelQuantizer::quantize_model_file("input.bin", "output.bin", QuantizationType::Q5_0) {
    Ok(()) => println!("Success!"),
    Err(QuantizeError::FileNotFound(path)) => {
        eprintln!("Model file not found: {}", path);
    }
    Err(QuantizeError::InvalidModel) => {
        eprintln!("Invalid model format");
    }
    Err(QuantizeError::QuantizationFailed(msg)) => {
        eprintln!("Quantization failed: {}", msg);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Advanced Usage

### Custom Quantization Parameters

```rust
use whisper_quantize::{ModelQuantizer, QuantizationType};
use std::path::Path;

// Check model compatibility before quantization
fn safe_quantize(input: &Path, output: &Path, qtype: QuantizationType) -> Result<(), Box<dyn std::error::Error>> {
    // Verify input exists
    if !input.exists() {
        return Err("Input file not found".into());
    }

    // Check if already quantized
    if let Some(existing_type) = ModelQuantizer::get_model_ftype(input)? {
        println!("Warning: Model already quantized as {}", existing_type);
    }

    // Estimate output size
    let input_size = std::fs::metadata(input)?.len();
    let estimated_output = (input_size as f64 * qtype.size_factor()) as u64;

    // Check available disk space
    // ... disk space check ...

    // Perform quantization
    ModelQuantizer::quantize_model_file(input, output, qtype)?;

    Ok(())
}
```

## Compatibility

- **Models**: All whisper.cpp compatible models (GGML format)
- **Platforms**: Windows, Linux, macOS (Intel & Apple Silicon)
- **Architectures**: x86_64, aarch64, arm
- **Rust**: 1.70.0 or later

## Benchmarks

Typical quantization speeds (on modern hardware):
- Tiny model (~39 MB): ~1 second
- Base model (~148 MB): ~3-5 seconds
- Small model (~466 MB): ~10-15 seconds
- Medium model (~1.5 GB): ~30-45 seconds
- Large model (~3.1 GB): ~60-90 seconds

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## See Also

- [whisper-cpp-rs](https://github.com/yourusername/whisper-cpp-rs) - Main transcription library
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - Original C++ implementation
- [Whisper](https://github.com/openai/whisper) - OpenAI's original Python implementation