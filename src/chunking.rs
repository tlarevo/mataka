//! Recursive character text splitter — ported from upstream Hindsight's
//! `fact_extraction.py` `_split_oversized_unit` (MIT License).
//!
//! Upstream uses langchain's `RecursiveCharacterTextSplitter`; we replicate
//! with plain Rust: try separators from coarsest to finest, splitting at the
//! last occurrence within the budget.

/// Separators ordered from coarsest to finest, matching upstream.
const SEPARATORS: &[&str] = &[
    "\n\n", // paragraph breaks
    "\n",   // line breaks
    ". ",   // sentence endings
    "! ",   // exclamations
    "? ",   // questions
    "; ",   // semicolons
    ", ",   // commas
    " ",    // words
    "",     // characters (last resort)
];

/// Split `text` into chunks of approximately `max_chars` characters with
/// `overlap_chars` of overlap between consecutive chunks.
///
/// - Empty or short text (≤ max_chars) returns a single chunk.
/// - Uses a character-based estimator (~4 chars/token) — see THA-136.
/// - Overlap is applied from the end of the previous chunk into the start of
///   the next (tail-prefix overlap), matching upstream behavior.
pub fn chunk_text(text: &str, max_chars: usize, overlap_chars: usize) -> Vec<String> {
    if text.len() <= max_chars {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        if remaining.len() <= max_chars {
            chunks.push(remaining.to_string());
            break;
        }

        // Try each separator from coarsest to finest
        let split_at = find_split_point(remaining, max_chars);
        let chunk = remaining[..split_at].to_string();
        chunks.push(chunk);

        // Advance with overlap
        let advance = if overlap_chars > 0 && split_at > overlap_chars {
            split_at - overlap_chars
        } else {
            split_at
        };
        // Ensure progress: at least one char forward
        let advance = advance.max(1);
        remaining = &remaining[advance..];
    }

    chunks
}

/// Find the best split point in `text` that stays within `max_chars`.
/// Tries each separator; returns the byte offset of the split.
fn find_split_point(text: &str, max_chars: usize) -> usize {
    for sep in SEPARATORS {
        if sep.is_empty() {
            // Last resort: split at max_chars boundary
            return max_chars;
        }
        // Find the last occurrence of `sep` within the first max_chars
        if let Some(idx) = text[..max_chars.min(text.len())].rfind(sep) {
            // Split right after the separator
            return idx + sep.len();
        }
    }
    max_chars
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_no_split() {
        let chunks = chunk_text("hello world", 1000, 100);
        assert_eq!(chunks, vec!["hello world"]);
    }

    #[test]
    fn splits_at_paragraph() {
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = chunk_text(text, 30, 0);
        assert!(chunks.len() > 1);
        // Each chunk should be ≤ max_chars (approx)
        for c in &chunks {
            assert!(c.len() <= 35); // small margin for separator
        }
    }

    #[test]
    fn overlap_produces_shared_content() {
        let text = "A".repeat(200);
        let chunks = chunk_text(&text, 80, 20);
        assert!(chunks.len() > 1);
        // Second chunk should start with content from end of first
        let end_of_first = chunks[0][chunks[0].len() - 20..].to_string();
        assert!(chunks[1].starts_with(&end_of_first));
    }

    #[test]
    fn handles_empty() {
        assert_eq!(chunk_text("", 1000, 100), vec![""]);
    }
}
