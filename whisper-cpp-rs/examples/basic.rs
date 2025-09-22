use std::path::Path;
use whisper_cpp_rs::{WhisperContext, FullParams, SamplingStrategy};
use hound;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check if model exists
    let model_path = "tests/models/ggml-tiny.en.bin";
    if !Path::new(model_path).exists() {
        eprintln!("Error: Model file not found at {}", model_path);
        eprintln!("Please download a model file first.");
        eprintln!("You can download the tiny.en model from:");
        eprintln!("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin");
        return Ok(());
    }

    println!("Loading Whisper model from {}...", model_path);
    let ctx = WhisperContext::new(model_path)?;

    println!("Model loaded successfully!");
    println!("Model info:");
    println!("  - Vocabulary size: {}", ctx.n_vocab());
    println!("  - Audio context: {}", ctx.n_audio_ctx());
    println!("  - Text context: {}", ctx.n_text_ctx());
    println!("  - Multilingual: {}", ctx.is_multilingual());

    // Load real audio for testing
    println!("\nLoading test audio...");
    let audio_path = "vendor/whisper.cpp/samples/jfk.wav";
    let audio = if Path::new(audio_path).exists() {
        println!("Loading audio from: {}", audio_path);
        load_wav_file(audio_path)?
    } else {
        eprintln!("Error: Audio file not found at {}", audio_path);
        eprintln!("Please ensure whisper.cpp repository is in vendor/ with sample files.");
        eprintln!("Or provide your own audio file at samples/test.wav");

        // Try alternative path
        let alt_path = "samples/test.wav";
        if Path::new(alt_path).exists() {
            println!("Loading alternative audio from: {}", alt_path);
            load_wav_file(alt_path)?
        } else {
            return Err(format!("No audio files found. Please provide audio at:\n  - {}\n  - {}", audio_path, alt_path).into());
        }
    };

    // Transcribe with default parameters
    println!("Transcribing with default parameters...");
    let text = ctx.transcribe(&audio)?;
    println!("Transcription result: '{}'", text);

    // Transcribe with custom parameters
    println!("\nTranscribing with custom parameters...");
    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
        .language("en")
        .translate(false)
        .no_timestamps(false)
        .temperature(0.0)
        .n_threads(2);

    let result = ctx.transcribe_with_full_params(&audio, params)?;
    println!("Full transcription result:");
    println!("  Text: '{}'", result.text);
    println!("  Segments: {}", result.segments.len());

    for (i, segment) in result.segments.iter().enumerate() {
        println!("    Segment {}: [{:.2}s - {:.2}s] '{}'",
            i + 1,
            segment.start_seconds(),
            segment.end_seconds(),
            segment.text
        );
    }

    println!("\nSuccess! The whisper.cpp Rust wrapper is working correctly.");

    Ok(())
}

fn load_wav_file(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    // Check format
    if spec.sample_rate != 16000 {
        eprintln!("Warning: Audio sample rate is {}Hz, expected 16000Hz", spec.sample_rate);
    }

    if spec.channels != 1 {
        eprintln!("Warning: Audio has {} channels, using first channel only", spec.channels);
    }

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .step_by(spec.channels as usize)
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    Ok(samples)
}