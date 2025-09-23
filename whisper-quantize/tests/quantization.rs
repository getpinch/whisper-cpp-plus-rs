//! Integration tests for model quantization functionality

use std::fs;
use std::path::Path;
use whisper_quantize::{ModelQuantizer, QuantizationType};

/// Helper to check if test models are available
fn test_models_available() -> bool {
    Path::new("tests/models/ggml-tiny.en.bin").exists()
}

#[test]
fn test_quantization_types() {
    // Test all quantization types are accessible
    let types = [
        QuantizationType::Q4_0,
        QuantizationType::Q4_1,
        QuantizationType::Q5_0,
        QuantizationType::Q5_1,
        QuantizationType::Q8_0,
        QuantizationType::Q2_K,
        QuantizationType::Q3_K,
        QuantizationType::Q4_K,
        QuantizationType::Q5_K,
        QuantizationType::Q6_K,
    ];

    for qtype in &types {
        // Verify name() works
        assert!(!qtype.name().is_empty());

        // Verify size_factor() returns reasonable values
        let factor = qtype.size_factor();
        assert!(factor > 0.0 && factor < 1.0,
            "{} has invalid size factor: {}", qtype, factor);
    }
}

#[test]
fn test_quantization_type_parsing() {
    // Test parsing from strings
    assert_eq!(QuantizationType::from_str("Q4_0"), Some(QuantizationType::Q4_0));
    assert_eq!(QuantizationType::from_str("q4_0"), Some(QuantizationType::Q4_0));
    assert_eq!(QuantizationType::from_str("Q40"), Some(QuantizationType::Q4_0));

    assert_eq!(QuantizationType::from_str("Q5_K"), Some(QuantizationType::Q5_K));
    assert_eq!(QuantizationType::from_str("q5k"), Some(QuantizationType::Q5_K));

    assert_eq!(QuantizationType::from_str("invalid"), None);
    assert_eq!(QuantizationType::from_str(""), None);
}

#[test]
fn test_quantization_display() {
    // Test Display implementation
    assert_eq!(format!("{}", QuantizationType::Q4_0), "Q4_0");
    assert_eq!(format!("{}", QuantizationType::Q5_K), "Q5_K");
}

#[test]
#[ignore] // Ignore by default as it requires model files
fn test_quantize_model() {
    if !test_models_available() {
        println!("Skipping test: model files not available");
        return;
    }

    let input_path = "tests/models/ggml-tiny.en.bin";
    let output_path = "tests/models/ggml-tiny.en-q5_0.bin";

    // Clean up any existing output file
    let _ = fs::remove_file(output_path);

    // Perform quantization
    let result = ModelQuantizer::quantize_model_file(
        input_path,
        output_path,
        QuantizationType::Q5_0,
    );

    assert!(result.is_ok(), "Quantization failed: {:?}", result);
    assert!(Path::new(output_path).exists(), "Output file was not created");

    // Verify the output file is smaller
    let input_size = fs::metadata(input_path).unwrap().len();
    let output_size = fs::metadata(output_path).unwrap().len();
    assert!(output_size < input_size,
        "Quantized model should be smaller: {} >= {}", output_size, input_size);

    // Clean up
    let _ = fs::remove_file(output_path);
}

#[test]
#[ignore] // Ignore by default as it requires model files
fn test_quantize_with_progress() {
    if !test_models_available() {
        println!("Skipping test: model files not available");
        return;
    }

    let input_path = "tests/models/ggml-tiny.en.bin";
    let output_path = "tests/models/ggml-tiny.en-q4_0.bin";

    // Clean up any existing output file
    let _ = fs::remove_file(output_path);

    // Track progress callbacks
    let mut progress_count = 0;
    let mut last_progress = 0.0;

    let result = ModelQuantizer::quantize_model_file_with_progress(
        input_path,
        output_path,
        QuantizationType::Q4_0,
        |progress| {
            progress_count += 1;
            assert!(progress >= last_progress, "Progress went backwards");
            assert!(progress >= 0.0 && progress <= 1.0, "Invalid progress value");
            last_progress = progress;
        },
    );

    assert!(result.is_ok(), "Quantization failed: {:?}", result);
    assert!(progress_count > 0, "Progress callback was never called");
    assert!(Path::new(output_path).exists(), "Output file was not created");

    // Clean up
    let _ = fs::remove_file(output_path);
}

#[test]
#[ignore] // Ignore by default as it requires model files
fn test_get_model_quantization_type() {
    if !test_models_available() {
        println!("Skipping test: model files not available");
        return;
    }

    let model_path = "tests/models/ggml-tiny.en.bin";

    // Check the original model (should be F16 or None for quantization type)
    let result = ModelQuantizer::get_model_quantization_type(model_path);
    assert!(result.is_ok(), "Failed to check model type: {:?}", result);

    match result.unwrap() {
        Some(qtype) => println!("Model is quantized as: {}", qtype),
        None => println!("Model is in full precision"),
    }
}

#[test]
#[ignore] // Ignore by default as it requires model files
fn test_estimate_quantized_size() {
    if !test_models_available() {
        println!("Skipping test: model files not available");
        return;
    }

    let model_path = "tests/models/ggml-tiny.en.bin";
    let original_size = fs::metadata(model_path).unwrap().len();

    // Test size estimation for different quantization types
    for qtype in QuantizationType::all() {
        let estimated = ModelQuantizer::estimate_quantized_size(model_path, *qtype).unwrap();

        // Estimated size should be smaller than original
        assert!(estimated < original_size,
            "{} estimation {} >= original {}", qtype, estimated, original_size);

        // Estimated size should be roughly size_factor * original
        let expected = (original_size as f64 * qtype.size_factor() as f64) as u64;
        let diff = if estimated > expected {
            estimated - expected
        } else {
            expected - estimated
        };

        // Allow 10% margin of error
        let margin = (expected as f64 * 0.1) as u64;
        assert!(diff < margin,
            "{}: estimated {} differs too much from expected {} (diff: {})",
            qtype, estimated, expected, diff);
    }
}

#[test]
#[ignore] // Ignore by default as it requires model files
fn test_context_quantize_method() {
    if !test_models_available() {
        println!("Skipping test: model files not available");
        return;
    }

    let input_path = "tests/models/ggml-tiny.en.bin";
    let output_path = "tests/models/ggml-tiny.en-q8_0.bin";

    // Clean up any existing output file
    let _ = fs::remove_file(output_path);

    // Use the convenience method on WhisperContext
    let result = WhisperContext::quantize_model(
        input_path,
        output_path,
        QuantizationType::Q8_0,
    );

    assert!(result.is_ok(), "Quantization via WhisperContext failed: {:?}", result);
    assert!(Path::new(output_path).exists(), "Output file was not created");

    // Try to load the quantized model to verify it's valid
    let load_result = WhisperContext::new(output_path);
    assert!(load_result.is_ok(), "Failed to load quantized model: {:?}", load_result);

    // Clean up
    let _ = fs::remove_file(output_path);
}

#[test]
fn test_error_handling() {
    // Test with non-existent input file
    let result = ModelQuantizer::quantize_model_file(
        "non_existent_model.bin",
        "output.bin",
        QuantizationType::Q4_0,
    );
    assert!(result.is_err(), "Should fail with non-existent input");

    // Test getting type of non-existent model
    let result = ModelQuantizer::get_model_quantization_type("non_existent.bin");
    assert!(result.is_err(), "Should fail with non-existent file");

    // Test size estimation of non-existent model
    let result = ModelQuantizer::estimate_quantized_size(
        "non_existent.bin",
        QuantizationType::Q5_0
    );
    assert!(result.is_err(), "Should fail with non-existent file");
}