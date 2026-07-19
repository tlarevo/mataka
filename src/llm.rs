//! LLM + embeddings over OpenAI-compatible HTTP.
//! Replaces Hindsight's llm_wrapper.py (litellm/torch/sentence-transformers stack).
//! Point base_url at OpenAI, Ollama (/v1), LM Studio, or any local minion server.
//! `mock` provider gives deterministic offline behavior for tests/dev.

use anyhow::{anyhow, Result};
use serde_json::{json, Value};

#[derive(Clone)]
pub struct LlmClient {
    pub provider: String, // openai-compatible | mock
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub embed_model: String,
    http: reqwest::Client,
}

impl LlmClient {
    /// Resolve config from env. Native `MATAKA_*` vars win; `HINDSIGHT_API_*`
    /// vars are accepted as drop-in fallbacks so existing Hindsight deployments
    /// work unchanged. Either/or — same knob, two names.
    pub fn from_env() -> Self {
        fn pick(mataka: &str, hindsight: &str, default: &str) -> String {
            std::env::var(mataka)
                .or_else(|_| std::env::var(hindsight))
                .unwrap_or_else(|_| default.to_string())
        }
        Self {
            provider: pick("MATAKA_LLM_PROVIDER", "HINDSIGHT_API_LLM_PROVIDER", "mock"),
            base_url: pick(
                "MATAKA_LLM_BASE_URL",
                "HINDSIGHT_API_LLM_BASE_URL",
                "http://localhost:11434/v1",
            ),
            api_key: pick("MATAKA_LLM_API_KEY", "HINDSIGHT_API_LLM_API_KEY", ""),
            model: pick("MATAKA_LLM_MODEL", "HINDSIGHT_API_LLM_MODEL", "qwen2.5:7b"),
            embed_model: pick(
                "MATAKA_EMBEDDINGS_MODEL",
                "HINDSIGHT_API_EMBEDDINGS_MODEL",
                "nomic-embed-text",
            ),
            http: reqwest::Client::new(),
        }
    }

    /// Chat completion returning raw text.
    pub async fn chat(&self, system: &str, user: &str, json_mode: bool) -> Result<String> {
        if self.provider == "mock" {
            return Ok(mock_chat(system, user));
        }
        let mut body = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0.1
        });
        if json_mode {
            body["response_format"] = json!({"type": "json_object"});
        }
        let resp: Value = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        resp["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow!("malformed chat response"))
    }

    /// Batch embeddings.
    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if self.provider == "mock" {
            return Ok(texts.iter().map(|t| mock_embed(t)).collect());
        }
        let resp: Value = self
            .http
            .post(format!("{}/embeddings", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&json!({"model": self.embed_model, "input": texts}))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let data = resp["data"]
            .as_array()
            .ok_or_else(|| anyhow!("bad embeddings response"))?;
        let mut out = Vec::with_capacity(data.len());
        for item in data {
            let v: Vec<f32> = item["embedding"]
                .as_array()
                .ok_or_else(|| anyhow!("bad embedding item"))?
                .iter()
                .filter_map(|x| x.as_f64().map(|f| f as f32))
                .collect();
            out.push(v);
        }
        Ok(out)
    }
}

/// Deterministic 64-dim hashing embedding: token hash buckets, L2-normalized.
/// Real semantic similarity is approximated by lexical overlap — good enough
/// to exercise the full pipeline offline.
fn mock_embed(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; 64];
    for tok in text.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
        if tok.is_empty() {
            continue;
        }
        let mut h: u64 = 1469598103934665603;
        for b in tok.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        v[(h % 64) as usize] += 1.0;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

/// Mock extraction: naive sentence split -> world facts; capitalized tokens -> entities.
fn mock_chat(system: &str, user: &str) -> String {
    if system.contains("fact extraction") {
        let facts: Vec<Value> = user
            .split(['.', '\n', '!', '?'])
            .map(str::trim)
            .filter(|s| s.len() > 8)
            .map(|s| {
                let entities: Vec<String> = s
                    .split_whitespace()
                    .filter(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) && w.len() > 2)
                    .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                    .filter(|w| !w.is_empty())
                    .collect();
                json!({"text": s, "fact_type": "world", "entities": entities, "occurred_start": null, "occurred_end": null})
            })
            .collect();
        json!({ "facts": facts }).to_string()
    } else {
        format!(
            "[mock reflect] Based on retained memories, regarding: {}",
            user.chars().take(120).collect::<String>()
        )
    }
}
