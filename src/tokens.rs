//! Exact token counting via tiktoken cl100k_base (GPT-4 / Embeddings).
//! Encoder is initialized once and cached for the process lifetime.

use std::sync::LazyLock;
use tiktoken_rs::cl100k_base;

static ENCODER: LazyLock<tiktoken_rs::CoreBPE> = LazyLock::new(|| {
    tracing::info!("initializing cl100k_base encoder (one-time)");
    cl100k_base().expect("cl100k_base encoding must be available")
});

/// Count tokens for `text` using the cl100k_base BPE encoder.
pub fn count(text: &str) -> usize {
    ENCODER.encode_ordinary(text).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_count() {
        assert!(count("hello world") > 0);
        assert!(count("") == 0);
    }

    #[test]
    fn longer_text_more_tokens() {
        let short = "The quick brown fox";
        let long = "The quick brown fox jumps over the lazy dog. It was a sunny day.";
        assert!(count(long) > count(short));
    }
}
