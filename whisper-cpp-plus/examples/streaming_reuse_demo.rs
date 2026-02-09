//! Demonstrates WhisperStream reuse across multiple sessions.

use std::path::Path;
use whisper_cpp_plus::{FullParams, SamplingStrategy, WhisperContext, WhisperStream, WhisperStreamConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "tests/models/ggml-tiny.en.bin";
    if !Path::new(model_path).exists() {
        eprintln!("Model not found at: {}", model_path);
        eprintln!("Showing API pattern instead:\n");
        show_pattern();
        return Ok(());
    }

    println!("Loading model...");
    let ctx = WhisperContext::new(model_path)?;

    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 })
        .language("en");

    let config = WhisperStreamConfig::default();
    let mut stream = WhisperStream::with_config(&ctx, params, config)?;

    for session in 1..=3 {
        println!("--- Session {} ---", session);

        if session > 1 {
            stream.reset();
            println!("  (reset — state reused)");
        }

        let audio = load_audio(2)?;
        stream.feed_audio(&audio);

        while let Some(segments) = stream.process_step()? {
            for seg in &segments {
                println!("  [{:.2}s-{:.2}s]: {}", seg.start_seconds(), seg.end_seconds(), seg.text);
            }
        }

        let flush_segs = stream.flush()?;
        for seg in &flush_segs {
            println!("  [flush {:.2}s-{:.2}s]: {}", seg.start_seconds(), seg.end_seconds(), seg.text);
        }

        println!("  processed: {} samples\n", stream.processed_samples());
    }

    println!("Done.");
    Ok(())
}

fn show_pattern() {
    println!("let mut stream = WhisperStream::new(&ctx, params)?;");
    println!("stream.feed_audio(&audio);");
    println!("while let Some(segs) = stream.process_step()? {{ ... }}");
    println!("stream.flush()?;");
    println!("stream.reset();  // reuse for next session");
}

fn load_audio(duration_secs: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Check env var first
    let path = if let Ok(dir) = std::env::var("WHISPER_TEST_AUDIO_DIR") {
        let p = format!("{}/jfk.wav", dir);
        if Path::new(&p).exists() { Some(p) } else { None }
    } else {
        None
    };

    let path = path.or_else(|| {
        let paths = [
            "../whisper-cpp-plus-sys/whisper.cpp/samples/jfk.wav",
            "whisper-cpp-plus-sys/whisper.cpp/samples/jfk.wav",
            "samples/audio.wav",
        ];
        paths.iter().find(|p| Path::new(p).exists()).map(|s| s.to_string())
    });

    let path = path.ok_or("No audio file found. Set WHISPER_TEST_AUDIO_DIR.")?;

    let mut reader = hound::WavReader::open(&path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .step_by(spec.channels as usize)
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    let needed = 16000 * duration_secs;
    Ok(samples.into_iter().take(needed).collect())
}
