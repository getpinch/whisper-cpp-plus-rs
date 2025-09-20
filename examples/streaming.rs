//! Example of streaming transcription
//!
//! This example demonstrates how to use WhisperStream for real-time
//! transcription of audio chunks as they arrive.

use std::path::Path;
use std::time::Duration;
use whisper_cpp_rs::{
    FullParams, SamplingStrategy, StreamConfig, StreamConfigBuilder, WhisperContext, WhisperStream,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check if model exists
    let model_path = "tests/models/ggml-tiny.en.bin";
    if !Path::new(model_path).exists() {
        eprintln!("Model file not found at: {}", model_path);
        eprintln!("Please download a model first. See README.md for instructions.");
        return Ok(());
    }

    println!("Loading Whisper model...");
    let context = WhisperContext::new(model_path)?;

    // Configure streaming parameters
    let stream_config = StreamConfigBuilder::new()
        .chunk_seconds(3.0)      // Process 3-second chunks
        .overlap_seconds(0.5)     // 0.5 second overlap between chunks
        .min_chunk_size(16000)    // Minimum 1 second before processing
        .partial_timeout(Duration::from_secs(2))
        .build();

    // Set up transcription parameters
    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
        .language("en")
        .no_timestamps(false)
        .print_progress(false);

    // Create the stream
    let mut stream = WhisperStream::with_config(&context, params, stream_config)?;

    println!("Streaming transcription initialized!");
    println!("Simulating real-time audio input...\n");

    // Simulate streaming audio input
    // In a real application, this would come from microphone, network, etc.
    let chunk_size = 16000; // 1 second chunks at 16kHz

    // Load sample audio (or generate silence for testing)
    let sample_audio = load_sample_audio()?;

    // Process audio in chunks to simulate streaming
    for (i, chunk) in sample_audio.chunks(chunk_size).enumerate() {
        // Simulate real-time delay
        std::thread::sleep(Duration::from_millis(500));

        println!("Feeding chunk {} ({} samples)...", i + 1, chunk.len());
        stream.feed_audio(chunk);

        // Process pending audio
        let segments = stream.process_pending()?;

        // Print new segments
        for segment in segments {
            println!(
                "[{:.2}s - {:.2}s]: {}",
                segment.start_seconds(),
                segment.end_seconds(),
                segment.text
            );
        }
    }

    // Flush any remaining audio
    println!("\nFlushing stream...");
    let final_segments = stream.flush()?;
    for segment in final_segments {
        println!(
            "[{:.2}s - {:.2}s]: {}",
            segment.start_seconds(),
            segment.end_seconds(),
            segment.text
        );
    }

    println!("\nTotal processed samples: {}", stream.processed_samples());
    println!("Streaming transcription complete!");

    Ok(())
}

/// Load sample audio for demonstration
fn load_sample_audio() -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Try to load the JFK sample if it exists
    let jfk_path = "vendor/whisper.cpp/samples/jfk.wav";

    if Path::new(jfk_path).exists() {
        println!("Loading JFK sample audio...");
        load_wav_16khz_mono(jfk_path)
    } else {
        // Generate simulated audio (silence with some noise)
        println!("Generating simulated audio (10 seconds)...");
        let mut audio = vec![0.0f32; 16000 * 10]; // 10 seconds

        // Add very slight noise to prevent complete silence
        for sample in audio.iter_mut() {
            *sample = (rand::random::<f32>() - 0.5) * 0.001;
        }

        Ok(audio)
    }
}

/// Load a WAV file and convert to 16kHz mono f32
fn load_wav_16khz_mono(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use hound;

    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    // Check sample rate
    if spec.sample_rate != 16000 {
        return Err(format!(
            "Audio must be 16kHz, but got {}Hz. Please resample the audio.",
            spec.sample_rate
        ).into());
    }

    // Convert samples to f32
    let samples: Result<Vec<f32>, _> = match spec.bits_per_sample {
        16 => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
            .collect(),
        32 => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
            .collect(),
        _ => {
            return Err(format!(
                "Unsupported bits per sample: {}",
                spec.bits_per_sample
            ).into())
        }
    };

    let samples = samples?;

    // Convert to mono if necessary
    let mono_samples = match spec.channels {
        1 => samples,
        2 => {
            // Average stereo channels to mono
            samples
                .chunks(2)
                .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                .collect()
        }
        _ => {
            return Err(format!(
                "Unsupported number of channels: {}",
                spec.channels
            ).into())
        }
    };

    Ok(mono_samples)
}

/// Add rand as a dev dependency for demo purposes
#[cfg(not(feature = "rand"))]
mod rand {
    pub fn random<T>() -> T
    where
        T: Default,
    {
        T::default()
    }
}