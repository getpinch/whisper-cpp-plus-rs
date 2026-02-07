//! Async streaming transcription example.
//!
//! Requires the `async` feature: `cargo run --example async_transcribe --features async`

#[cfg(feature = "async")]
#[tokio::main]
async fn main() -> whisper_cpp_plus::Result<()> {
    use whisper_cpp_plus::{
        AsyncWhisperStream, FullParams, SamplingStrategy, WhisperContext, WhisperStreamConfig,
    };

    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/models/ggml-tiny.en.bin".to_string());

    let ctx = WhisperContext::new(&model_path)?;
    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 }).language("en");
    let config = WhisperStreamConfig::default();

    let mut stream = AsyncWhisperStream::with_config(ctx, params, config)?;

    // Feed 3 seconds of silence (enough for one fixed step)
    let audio = vec![0.0f32; 48000];
    stream.feed_audio(audio).await?;

    // Receive segments
    if let Some(segments) = stream.recv_segments().await {
        for seg in &segments {
            println!("[{} - {}] {}", seg.start_ms, seg.end_ms, seg.text);
        }
    }

    stream.stop().await?;
    Ok(())
}

#[cfg(not(feature = "async"))]
fn main() {
    eprintln!("This example requires the `async` feature:");
    eprintln!("  cargo run --example async_transcribe --features async");
}
