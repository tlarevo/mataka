//! Memory vetting engine — deterministic, non-LLM, offline.
//! Detects secrets and sensitive data in text using compiled regex rules
//! (ported from gitleaks/ripsecrets) and Shannon-entropy analysis.
//!
//! Three call sites share one engine:
//!   1. Ingress guard (retain pipeline, before embedding)
//!   2. At-rest scanner (CLI/API, scans existing data)
//!   3. Egress guard (optional belt-and-suspenders on recall results)

use anyhow::Result;
use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;

// ─── Rule definitions (embedded at compile time) ───────────────────────

const RULES_TOML: &str = include_str!("../rules/vet.toml");

#[derive(Debug, Deserialize)]
struct RulesFile {
    version: u32,
    rules: Vec<RuleDef>,
}

#[derive(Debug, Deserialize)]
struct RuleDef {
    id: String,
    #[allow(dead_code)]
    description: String,
    pattern: String,
    placeholder: String,
}

// ─── Compiled rule ─────────────────────────────────────────────────────

pub struct Rule {
    pub id: String,
    pub placeholder: String,
    re: Regex,
}

// ─── Detection result ──────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct Detection {
    pub rule_id: String,
    pub count: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ScanResult {
    pub detections: Vec<Detection>,
    pub redacted_text: String,
}

// ─── Vet mode ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VetMode {
    /// Redact detected secrets in-place (default)
    Redact,
    /// Reject the entire retain item with 422
    Strict,
    /// Vetting disabled
    Off,
}

impl VetMode {
    pub fn from_env() -> Self {
        match std::env::var("MATAKA_VET").as_deref() {
            Ok("strict") => VetMode::Strict,
            Ok("off") => VetMode::Off,
            _ => VetMode::Redact,
        }
    }
}

// ─── Shannon entropy ───────────────────────────────────────────────────

/// Shannon entropy threshold. Bits per char — tokens above this are
/// likely secrets, not natural language. Calibrated against:
/// - git SHAs (40 hex chars → 4.0 bits/char)
/// - UUIDs (36 chars → ~3.8 bits/char)
/// - base64 content hashes → ~5.5 bits/char
///
/// Threshold of 4.5 catches most API keys (typically 5+ bits/char)
/// while letting git SHAs and UUIDs pass.
const ENTROPY_THRESHOLD: f64 = 4.5;

fn shannon_entropy(s: &str) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    let mut freq = HashMap::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            *freq.entry(c).or_insert(0u64) += 1;
        }
    }
    let len = freq.values().sum::<u64>() as f64;
    if len == 0.0 {
        return 0.0;
    }
    let entropy: f64 = freq
        .values()
        .map(|&count| {
            let p = count as f64 / len;
            -p * (p.ln() / std::f64::consts::LN_2)
        })
        .sum();
    entropy
}

// ─── Engine ────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub struct VetEngine {
    rules: Vec<Rule>,
    version: u32,
}

impl VetEngine {
    /// Build the engine from the embedded rules TOML.
    pub fn new() -> Result<Self> {
        let parsed: RulesFile = toml::from_str(RULES_TOML)?;
        let mut rules = Vec::with_capacity(parsed.rules.len());
        for def in &parsed.rules {
            let re = Regex::new(&def.pattern)?;
            rules.push(Rule {
                id: def.id.clone(),
                placeholder: def.placeholder.clone(),
                re,
            });
        }
        Ok(Self {
            rules,
            version: parsed.version,
        })
    }

    #[allow(dead_code)]
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Scan text for secrets. Returns detections + redacted text.
    /// Never logs or stores the matched secret — rule id + position only.
    pub fn scan(&self, text: &str) -> ScanResult {
        let mut redacted = text.to_string();
        let mut detections: Vec<Detection> = Vec::new();
        let mut seen_rules: HashMap<String, usize> = HashMap::new();

        // Phase 1: regex rule matching
        for rule in &self.rules {
            let matches: Vec<_> = rule
                .re
                .find_iter(&redacted)
                .map(|m| (m.start(), m.end()))
                .collect();
            if !matches.is_empty() {
                let count = matches.len();
                // Replace from end to preserve indices
                for &(start, end) in matches.iter().rev() {
                    redacted.replace_range(start..end, &rule.placeholder);
                }
                *seen_rules.entry(rule.id.clone()).or_insert(0) += count;
            }
        }

        // Phase 2: entropy check on remaining unrecognized tokens
        // Extract alphanumeric tokens that weren't caught by regex rules
        let token_re = Regex::new(r"[A-Za-z0-9/+=_-]{20,}").unwrap();
        let mut entropy_hits = 0usize;
        let token_matches: Vec<_> = token_re
            .find_iter(&redacted)
            .filter(|m| !m.as_str().contains("{{"))
            .map(|m| (m.start(), m.end(), m.as_str().to_string()))
            .collect();
        for (start, end, token) in token_matches {
            if shannon_entropy(&token) > ENTROPY_THRESHOLD {
                entropy_hits += 1;
                redacted.replace_range(start..end, "{{REDACTED:high_entropy_token}}");
            }
        }
        if entropy_hits > 0 {
            *seen_rules.entry("entropy_high".into()).or_insert(0) += entropy_hits;
        }

        // Build detections list
        for (rule_id, count) in seen_rules {
            detections.push(Detection { rule_id, count });
        }
        detections.sort_by(|a, b| a.rule_id.cmp(&b.rule_id));

        ScanResult {
            detections,
            redacted_text: redacted,
        }
    }

    /// Ingress guard: scan text, return redacted version.
    /// Returns Err if mode is Strict and secrets were found.
    pub fn vet_text(&self, text: &str, mode: VetMode) -> Result<(String, Vec<Detection>)> {
        if mode == VetMode::Off {
            return Ok((text.to_string(), vec![]));
        }
        let result = self.scan(text);
        if !result.detections.is_empty() && mode == VetMode::Strict {
            let rule_ids: Vec<&str> = result
                .detections
                .iter()
                .map(|d| d.rule_id.as_str())
                .collect();
            anyhow::bail!(
                "Vetting rejected: secrets detected [{}]",
                rule_ids.join(", ")
            );
        }
        Ok((result.redacted_text, result.detections))
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_loads() {
        let engine = VetEngine::new().expect("rules/vet.toml failed to parse");
        assert!(engine.version() >= 1);
        assert!(!engine.rules.is_empty());
    }

    #[test]
    fn detects_aws_access_key() {
        let engine = VetEngine::new().unwrap();
        let text = "Use AKIAIOSFODNN7EXAMPLE for the AWS key";
        let result = engine.scan(text);
        assert!(result
            .detections
            .iter()
            .any(|d| d.rule_id == "aws_access_key"));
        assert!(result.redacted_text.contains("{{REDACTED:aws_access_key}}"));
        assert!(!result.redacted_text.contains("AKIAIOSFODNN7EXAMPLE"));
    }

    #[test]
    fn detects_github_token() {
        let engine = VetEngine::new().unwrap();
        let text = "Token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij";
        let result = engine.scan(text);
        assert!(result
            .detections
            .iter()
            .any(|d| d.rule_id == "github_token"));
        assert!(result.redacted_text.contains("{{REDACTED:github_token}}"));
    }

    #[test]
    fn detects_slack_token() {
        let engine = VetEngine::new().unwrap();
        let text = "Bot token: xoxb-1234567890-1234567890123-AbCdEfGhIjKlMnOpQrStUv";
        let result = engine.scan(text);
        assert!(result.detections.iter().any(|d| d.rule_id == "slack_token"));
        assert!(result.redacted_text.contains("{{REDACTED:slack_token}}"));
    }

    #[test]
    fn detects_private_key() {
        let engine = VetEngine::new().unwrap();
        let text =
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----";
        let result = engine.scan(text);
        assert!(result
            .detections
            .iter()
            .any(|d| d.rule_id == "private_key_block"));
        assert!(result.redacted_text.contains("{{REDACTED:private_key}}"));
    }

    #[test]
    fn detects_connection_string() {
        let engine = VetEngine::new().unwrap();
        let text = "Connect to postgres://admin:secret123@db.example.com:5432/mydb";
        let result = engine.scan(text);
        assert!(result
            .detections
            .iter()
            .any(|d| d.rule_id == "connection_string"));
    }

    #[test]
    fn detects_openai_key() {
        let engine = VetEngine::new().unwrap();
        let text = "Key: sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh";
        let result = engine.scan(text);
        assert!(result
            .detections
            .iter()
            .any(|d| d.rule_id == "openai_key_short"));
    }

    #[test]
    fn detects_anthropic_key() {
        let engine = VetEngine::new().unwrap();
        let text = "ANTHROPIC_KEY=sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let result = engine.scan(text);
        assert!(result
            .detections
            .iter()
            .any(|d| d.rule_id == "anthropic_key"));
    }

    #[test]
    fn detects_jwt() {
        let engine = VetEngine::new().unwrap();
        let text = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U";
        let result = engine.scan(text);
        assert!(result.detections.iter().any(|d| d.rule_id == "jwt"));
    }

    #[test]
    fn benign_lookalikes_not_flagged() {
        let engine = VetEngine::new().unwrap();
        // Git SHA, UUID, base64 hash — all should pass
        let text = "Commit a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2, \
                     UUID 550e8400-e29b-41d4-a716-446655440000, \
                     hash dGVzdA==";
        let result = engine.scan(text);
        assert!(
            result.detections.is_empty(),
            "benign lookalikes flagged: {:?}",
            result.detections
        );
    }

    #[test]
    fn high_entropy_detection() {
        let engine = VetEngine::new().unwrap();
        // A truly random-looking string (>20 chars, high entropy)
        let text = "secret is kJ9sB2mN4pQ7rT1vX3yZ5aC8eG2hJ4kL6nP";
        let result = engine.scan(text);
        assert!(
            result
                .detections
                .iter()
                .any(|d| d.rule_id == "entropy_high"),
            "high-entropy token not caught: {:?}",
            result.detections
        );
    }

    #[test]
    fn multiple_secrets_in_one_text() {
        let engine = VetEngine::new().unwrap();
        let text = "AWS: AKIAIOSFODNN7EXAMPLE\nGitHub: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij";
        let result = engine.scan(text);
        assert!(result.detections.len() >= 2);
        assert!(!result.redacted_text.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(!result
            .redacted_text
            .contains("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"));
    }

    #[test]
    fn strict_mode_rejects() {
        let engine = VetEngine::new().unwrap();
        let text = "Key is AKIAIOSFODNN7EXAMPLE";
        let err = engine.vet_text(text, VetMode::Strict).unwrap_err();
        assert!(err.to_string().contains("aws_access_key"));
    }

    #[test]
    fn off_mode_passes_through() {
        let engine = VetEngine::new().unwrap();
        let text = "Key is AKIAIOSFODNN7EXAMPLE";
        let (out, detections) = engine.vet_text(text, VetMode::Off).unwrap();
        assert_eq!(out, text);
        assert!(detections.is_empty());
    }

    #[test]
    fn redact_mode_produces_typed_placeholder() {
        let engine = VetEngine::new().unwrap();
        let text = "Set AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
        let (out, detections) = engine.vet_text(text, VetMode::Redact).unwrap();
        assert!(!detections.is_empty());
        assert!(out.contains("{{REDACTED:"));
        assert!(!out.contains("wJalrXUtnFEMI"));
    }

    #[test]
    fn env_secret_assignment_detected() {
        let engine = VetEngine::new().unwrap();
        let text = "API_KEY=super_secret_value_here_12345";
        let result = engine.scan(text);
        assert!(result
            .detections
            .iter()
            .any(|d| d.rule_id == "env_secret_assignment"));
    }

    #[test]
    fn detection_count_accurate() {
        let engine = VetEngine::new().unwrap();
        let text = "AKIA1111111111111111 AKIA2222222222222222 AKIA3333333333333333";
        let result = engine.scan(text);
        let aws = result
            .detections
            .iter()
            .find(|d| d.rule_id == "aws_access_key")
            .unwrap();
        assert_eq!(aws.count, 3);
    }
}
