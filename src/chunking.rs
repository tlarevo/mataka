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

        // Advance with overlap, floored to a char boundary so `remaining[advance..]`
        // can't panic when `max_chars`/`overlap_chars` land mid-codepoint (THA: was
        // a repeatable panic on any text with multi-byte UTF-8 — em dashes, emoji —
        // near the chunk boundary).
        let advance = if overlap_chars > 0 && split_at > overlap_chars {
            split_at - overlap_chars
        } else {
            split_at
        };
        let advance = floor_char_boundary(remaining, advance.max(1));
        // Ensure progress: floor_char_boundary can floor to 0 when the first char
        // is wider than `advance` bytes — step over that whole char instead.
        let advance = if advance == 0 {
            remaining.chars().next().map(char::len_utf8).unwrap_or(1)
        } else {
            advance
        };
        remaining = &remaining[advance..];
    }

    chunks
}

/// Round `idx` down to the nearest UTF-8 char boundary of `text`. O(1): a char
/// is at most 4 bytes, so this loops at most 3 times.
fn floor_char_boundary(text: &str, idx: usize) -> usize {
    let mut idx = idx.min(text.len());
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Find the best split point in `text` that stays within `max_chars`.
/// Tries each separator; returns the byte offset of the split.
fn find_split_point(text: &str, max_chars: usize) -> usize {
    let limit = floor_char_boundary(text, max_chars);
    for sep in SEPARATORS {
        if sep.is_empty() {
            // Last resort: split at max_chars boundary
            return limit;
        }
        // Find the last occurrence of `sep` within the first max_chars
        if let Some(idx) = text[..limit].rfind(sep) {
            // Split right after the separator (ascii separator, always a boundary)
            return idx + sep.len();
        }
    }
    limit
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

    /// Regression for a production panic: multi-byte UTF-8 chars (em dash,
    /// emoji) landing exactly on a byte-offset boundary used to panic both
    /// `find_split_point`'s slice and `chunk_text`'s advance-by-overlap slice.
    #[test]
    fn no_panic_on_multibyte_boundary() {
        // '—' is 3 bytes; padded so max_chars=8000 lands inside it, matching
        // the exact production repro (chunking.rs:70 panic).
        let mut text = "a".repeat(7999);
        text.push('—');
        text.push_str(&"b".repeat(500));
        let chunks = chunk_text(&text, 8000, 800);
        assert!(!chunks.is_empty());
        // Overlap duplicates chars by design, so just check no data is lost:
        // first/last chunk cover the text's start/end.
        assert!(text.starts_with(chunks.first().unwrap().chars().next().unwrap()));
        assert!(text.ends_with(chunks.last().unwrap().chars().last().unwrap()));

        // Emoji (4 bytes) variant, mirroring the '✅' production panic.
        let mut text2 = "x".repeat(7998);
        text2.push('✅');
        text2.push_str(&"y".repeat(500));
        let chunks2 = chunk_text(&text2, 8000, 800);
        assert!(!chunks2.is_empty());
    }
}
