//! Example program to quantize Whisper models
//!
//! Usage: cargo run --example quantize_model <input_model> <output_model> <quantization_type>
//!
//! Quantization types: q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k, q4_k, q5_k, q6_k

use whisper_quantize::{ModelQuantizer, QuantizationType};
use std::env;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        eprintln!("Usage: {} <input_model> <output_model> <quantization_type>", args[0]);
        eprintln!();
        eprintln!("Quantization types:");
        eprintln!("  q4_0 - 4-bit quantization (method 0)");
        eprintln!("  q4_1 - 4-bit quantization (method 1)");
        eprintln!("  q5_0 - 5-bit quantization (method 0)");
        eprintln!("  q5_1 - 5-bit quantization (method 1)");
        eprintln!("  q8_0 - 8-bit quantization");
        eprintln!("  q2_k - 2-bit k-quantization");
        eprintln!("  q3_k - 3-bit k-quantization");
        eprintln!("  q4_k - 4-bit k-quantization");
        eprintln!("  q5_k - 5-bit k-quantization");
        eprintln!("  q6_k - 6-bit k-quantization");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let qtype_str = &args[3];

    // Parse quantization type
    let qtype = QuantizationType::from_str(qtype_str)
        .ok_or_else(|| format!("Invalid quantization type: {}", qtype_str))?;

    // Check if input file exists
    if !Path::new(input_path).exists() {
        eprintln!("Error: Input file does not exist: {}", input_path);
        std::process::exit(1);
    }

    // Get file size for displaying size reduction
    let input_size = std::fs::metadata(input_path)?.len();
    let input_size_mb = input_size as f64 / (1024.0 * 1024.0);

    println!("Quantizing model:");
    println!("  Input:  {} ({:.2} MB)", input_path, input_size_mb);
    println!("  Output: {}", output_path);
    println!("  Type:   {} ({})", qtype, qtype.name());
    println!();

    // Perform quantization with progress callback
    ModelQuantizer::quantize_model_file_with_progress(
        input_path,
        output_path,
        qtype,
        |progress| {
            print!("\rProgress: {:.1}%", progress * 100.0);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
    )?;

    println!("\n");

    // Display results
    if Path::new(output_path).exists() {
        let output_size = std::fs::metadata(output_path)?.len();
        let output_size_mb = output_size as f64 / (1024.0 * 1024.0);
        let reduction = (1.0 - (output_size as f64 / input_size as f64)) * 100.0;

        println!("Quantization complete!");
        println!("  Output size: {:.2} MB", output_size_mb);
        println!("  Size reduction: {:.1}%", reduction);
        println!("  Compression ratio: {:.2}x", input_size as f64 / output_size as f64);
    } else {
        eprintln!("Error: Output file was not created");
        std::process::exit(1);
    }

    Ok(())
}