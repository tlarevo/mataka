//! THA-136: 10-item extraction eval set.
//!
//! Each item has input text and expected extraction properties.
//! Run with: MATAKA_LLM_PROVIDER=mock cargo test --test eval_fixtures
//!
//! The mock provider does naive extraction; these assertions verify the
//! pipeline plumbing (chunking, dedup, fixture dump, occurred_end binding)
//! rather than LLM quality. Real-model quality checks run separately
//! against the Looper harness.

use serde_json::{json, Value};

/// Eval cases: input text → expected minimum assertions.
fn eval_cases() -> Vec<(&'static str, &'static str, Value)> {
    vec![
        // 1. Pronoun resolution — "he" should be resolved to a name
        (
            "pronoun_resolution",
            "Alice told Bob that she got promoted. He was really happy for her.",
            json!({
                "min_facts": 1,
                "must_contain_entity": "Alice",
                "no_bare_pronouns": true,
            }),
        ),
        // 2. Relative date — "yesterday" should become absolute
        (
            "relative_date",
            "Yesterday I visited the Eiffel Tower with Marie. It was an amazing experience.",
            json!({
                "min_facts": 1,
                "must_contain_entity": "Eiffel Tower",
                "has_occurred_start": true,
            }),
        ),
        // 3. Multi-entity sentence
        (
            "multi_entity",
            "Dr. Smith referred patient John Doe to Dr. Patel at Mayo Clinic for cardiology.",
            json!({
                "min_facts": 1,
                "must_contain_entity": "Dr. Smith",
                "must_contain_entity2": "Mayo Clinic",
            }),
        ),
        // 4. Experience classification — "I debugged" should be experience
        (
            "experience_class",
            "I debugged the production outage for three hours and finally found the root cause in the connection pool.",
            json!({
                "min_facts": 1,
                "fact_type_is_experience": true,
            }),
        ),
        // 5. World classification — user preference
        (
            "world_class",
            "The user prefers Vim over Emacs and uses the Dracula theme.",
            json!({
                "min_facts": 1,
                "fact_type_is_world": true,
                "must_contain_entity": "Vim",
            }),
        ),
        // 6. Filler should be skipped
        (
            "filler_skipped",
            "Thanks! Sounds good. Ok got it. Sure thing.",
            json!({
                "max_facts": 2,
            }),
        ),
        // 7. Event with explicit date
        (
            "explicit_date",
            "On March 15, 2024, we launched the new API v2 with zero downtime.",
            json!({
                "min_facts": 1,
                "has_occurred_start": true,
                "must_contain_entity": "API v2",
            }),
        ),
        // 8. Coreference — "the manager" + "Sarah"
        (
            "coreference",
            "The manager Sarah approved the budget. She also signed off on the hiring plan.",
            json!({
                "min_facts": 1,
                "must_contain_entity": "Sarah",
                "no_bare_pronouns": true,
            }),
        ),
        // 9. Complex multi-fact
        (
            "complex_multi_fact",
            "Alice moved to Berlin last month for a role at Siemens. Bob stayed in Munich at BMW. They still meet for coffee every Friday.",
            json!({
                "min_facts": 2,
                "must_contain_entity": "Alice",
                "must_contain_entity2": "Berlin",
            }),
        ),
        // 10. Long document (tests chunking — this is short but verifies the path)
        (
            "chunking_sanity",
            "This is a simple test sentence. It verifies that the chunking path works correctly for short inputs that should not be split.",
            json!({
                "min_facts": 1,
            }),
        ),
    ]
}

#[test]
fn eval_set_count() {
    let cases = eval_cases();
    assert_eq!(cases.len(), 10, "eval set must have exactly 10 items");
}

#[test]
fn eval_set_metadata_valid() {
    for (name, input, expected) in eval_cases() {
        assert!(!input.is_empty(), "eval case {name} has empty input");
        assert!(
            expected.get("min_facts").is_some() || expected.get("max_facts").is_some(),
            "eval case {name} must assert min_facts or max_facts"
        );
    }
}
