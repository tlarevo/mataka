//! LLM + embeddings over OpenAI-compatible HTTP.
//! Replaces Hindsight's llm_wrapper.py (litellm/torch/sentence-transformers stack).
//! Point base_url at OpenAI, Ollama (/v1), LM Studio, or any local minion server.
//! `mock` provider gives deterministic offline behavior for tests/dev.

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

/// Bounded retries for transient failures only (busy/timeout) — never masks a
/// wedged server indefinitely (see mataka incident: Ollama's queue silently
/// stuck with 0 resident models for days, every /v1/* call 503ing).
const MAX_ATTEMPTS: u32 = 4;
const INITIAL_BACKOFF: Duration = Duration::from_millis(500);
/// `/health` probe budget — short so a health check never hangs or competes
/// with real generation for long.
const HEALTH_PROBE_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Clone)]
pub struct LlmClient {
    pub provider: String, // openai-compatible | mock
    /// Embeddings endpoint (and chat's fallback when `chat_base_url` is unset).
    pub base_url: String,
    /// Chat/completions endpoint. Defaults to `base_url` — set
    /// `MATAKA_LLM_CHAT_BASE_URL` to route chat to a different server than
    /// embeddings (e.g. a chat-only local model that has no embeddings
    /// endpoint at all, like apfel or turbo-fieldfare).
    pub chat_base_url: String,
    pub api_key: String,
    pub model: String,
    pub embed_model: String,
    http: reqwest::Client,
    /// Caps in-flight requests so a burst can't queue hundreds deep against a
    /// local single-model server (e.g. a small dedicated inference runtime).
    concurrency: Arc<Semaphore>,
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
        let timeout_secs: u64 = std::env::var("MATAKA_LLM_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(120);
        let max_concurrent: usize = std::env::var("MATAKA_LLM_MAX_CONCURRENT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        let base_url = pick(
            "MATAKA_LLM_BASE_URL",
            "HINDSIGHT_API_LLM_BASE_URL",
            "http://localhost:11434/v1",
        );
        let chat_base_url = std::env::var("MATAKA_LLM_CHAT_BASE_URL")
            .ok()
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| base_url.clone());
        Self {
            provider: pick("MATAKA_LLM_PROVIDER", "HINDSIGHT_API_LLM_PROVIDER", "mock"),
            base_url,
            chat_base_url,
            api_key: pick("MATAKA_LLM_API_KEY", "HINDSIGHT_API_LLM_API_KEY", ""),
            model: pick("MATAKA_LLM_MODEL", "HINDSIGHT_API_LLM_MODEL", "qwen2.5:7b"),
            embed_model: pick(
                "MATAKA_EMBEDDINGS_MODEL",
                "HINDSIGHT_API_EMBEDDINGS_MODEL",
                "nomic-embed-text",
            ),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(timeout_secs))
                .build()
                .expect("failed to build reqwest client"),
            concurrency: Arc::new(Semaphore::new(max_concurrent.max(1))),
        }
    }

    /// POST `path` against `base_url` with bounded retry on 503 (busy) and
    /// request timeouts. Holds a concurrency permit for the whole attempt
    /// loop so retries don't bypass the cap. Any other status/error fails
    /// immediately — this must never turn into an unbounded retry loop
    /// against a genuinely wedged server (that's what silently broke mataka
    /// against Ollama).
    async fn post_json(&self, base_url: &str, path: &str, body: &Value) -> Result<Value> {
        let _permit = self
            .concurrency
            .acquire()
            .await
            .expect("semaphore never closed");
        let mut backoff = INITIAL_BACKOFF;
        for attempt in 1..=MAX_ATTEMPTS {
            let outcome = self
                .http
                .post(format!("{base_url}{path}"))
                .bearer_auth(&self.api_key)
                .json(body)
                .send()
                .await;
            match outcome {
                Ok(resp) if resp.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE => {
                    if attempt == MAX_ATTEMPTS {
                        return Err(anyhow!(
                            "llm request to {path} still 503 after {MAX_ATTEMPTS} attempts"
                        ));
                    }
                }
                Ok(resp) => return Ok(resp.error_for_status()?.json().await?),
                Err(e) if e.is_timeout() && attempt < MAX_ATTEMPTS => {}
                Err(e) => return Err(e.into()),
            }
            tokio::time::sleep(backoff).await;
            backoff *= 2;
        }
        unreachable!("loop returns or errors on its final attempt")
    }

    /// Chat completion returning raw text.
    pub async fn chat(&self, system: &str, user: &str, json_mode: bool) -> Result<String> {
        if self.provider == "mock" {
            if std::env::var("MATAKA_MOCK_FAIL").is_ok() {
                return Err(anyhow::anyhow!("mock failure (MATAKA_MOCK_FAIL set)"));
            }
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
        let resp = self
            .post_json(&self.chat_base_url, "/chat/completions", &body)
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
        let resp = self
            .post_json(
                &self.base_url,
                "/embeddings",
                &json!({"model": self.embed_model, "input": texts}),
            )
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

    /// Fast LLM liveness probe for `/health`. Hits the embeddings path — the
    /// one `retain` and `recall` both depend on — so a wedged generation queue
    /// (ollama up, `/api/tags` fine, `/v1/*` dead) surfaces as unreachable.
    /// Deliberately bypasses the concurrency semaphore and uses a short
    /// timeout so health checks never block real work or hold the cap.
    pub async fn check_llm(&self) -> Result<Duration> {
        if self.provider == "mock" {
            return Ok(Duration::ZERO);
        }
        let t0 = std::time::Instant::now();
        tokio::time::timeout(HEALTH_PROBE_TIMEOUT, async {
            self.http
                .post(format!("{}/embeddings", self.base_url))
                .bearer_auth(&self.api_key)
                .json(&json!({"model": self.embed_model, "input": ["health"]}))
                .send()
                .await?
                .error_for_status()?;
            Ok::<_, anyhow::Error>(())
        })
        .await
        .map_err(|_| anyhow!("llm health probe timed out after {HEALTH_PROBE_TIMEOUT:?}"))??;
        Ok(t0.elapsed())
    }
}

/// Deterministic 64-dim hashing embedding: token hash buckets, L2-normalized.
/// Real semantic similarity is approximated by lexical overlap — good enough
/// to exercise the full pipeline offline.
pub(crate) fn mock_embed(text: &str) -> Vec<f32> {
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
    } else if system.contains("consolidation") {
        // Mock consolidation: parse source facts from user message, concatenate deduplicated sentences
        let source_ids: Vec<String> = user
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                if let Some(id_start) = line.find("\"id\":") {
                    let rest = &line[id_start + 5..].trim();
                    let id: String = rest
                        .trim_matches(|c: char| c == '"' || c == ',' || c == ' ')
                        .to_string();
                    Some(id)
                } else {
                    None
                }
            })
            .collect();
        let texts: Vec<String> = user
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                if let Some(text_start) = line.find("\"text\":") {
                    let rest = &line[text_start + 7..].trim();
                    let text: String = rest
                        .trim_matches(|c: char| c == '"' || c == ',')
                        .to_string();
                    Some(text)
                } else {
                    None
                }
            })
            .collect();
        // Dedupe sentences
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        let deduped: Vec<&str> = texts
            .iter()
            .flat_map(|t| t.split(". "))
            .filter(|s| !s.is_empty() && seen.insert(s.to_lowercase()))
            .collect();
        let observation = deduped.join(". ");
        json!({
            "observation_text": observation,
            "source_ids": source_ids,
            "supersedes": true
        })
        .to_string()
    } else {
        format!(
            "[mock reflect] Based on retained memories, regarding: {}",
            user.chars().take(120).collect::<String>()
        )
    }
}

#[cfg(test)]
mod resilience_tests {
    use super::*;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use axum::routing::post;
    use axum::{Json, Router};
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Spins up a real HTTP server on a random port that returns 503 for the
    /// first `fail_times` requests to `/chat/completions`, then 200.
    async fn spawn_flaky_server(fail_times: u32) -> (String, Arc<AtomicU32>) {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_for_route = counter.clone();
        let app = Router::new().route(
            "/chat/completions",
            post(move |Json(_body): Json<Value>| {
                let counter = counter_for_route.clone();
                async move {
                    let n = counter.fetch_add(1, Ordering::SeqCst);
                    if n < fail_times {
                        StatusCode::SERVICE_UNAVAILABLE.into_response()
                    } else {
                        Json(json!({"choices":[{"message":{"content":"ok after retry"}}]}))
                            .into_response()
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}"), counter)
    }

    fn client_for(base_url: String) -> LlmClient {
        LlmClient {
            provider: "openai-compatible".into(),
            chat_base_url: base_url.clone(),
            base_url,
            api_key: String::new(),
            model: "test-model".into(),
            embed_model: "test-embed".into(),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap(),
            concurrency: Arc::new(Semaphore::new(4)),
        }
    }

    #[tokio::test]
    async fn chat_retries_on_503_then_succeeds() {
        let (base_url, _counter) = spawn_flaky_server(2).await;
        let result = client_for(base_url).chat("sys", "user", false).await;
        assert_eq!(result.unwrap(), "ok after retry");
    }

    #[tokio::test]
    async fn chat_gives_up_after_max_attempts_instead_of_looping_forever() {
        let (base_url, counter) = spawn_flaky_server(u32::MAX).await;
        let result = client_for(base_url).chat("sys", "user", false).await;
        assert!(result.is_err(), "must surface the error, not hang");
        assert_eq!(counter.load(Ordering::SeqCst), MAX_ATTEMPTS);
    }

    #[tokio::test]
    async fn chat_and_embed_route_to_independently_configured_servers() {
        // Two separate servers: chat-only (embed unimplemented, matching
        // apfel/turbo-fieldfare) and embed-only. Proves MATAKA_LLM_CHAT_BASE_URL
        // actually decouples chat from the embeddings base_url.
        let (chat_url, _c1) = spawn_flaky_server(0).await;
        let embed_url = spawn_embed_server().await;

        let client = LlmClient {
            provider: "openai-compatible".into(),
            chat_base_url: chat_url,
            base_url: embed_url,
            api_key: String::new(),
            model: "chat-model".into(),
            embed_model: "embed-model".into(),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap(),
            concurrency: Arc::new(Semaphore::new(4)),
        };

        let chat_result = client.chat("sys", "user", false).await.unwrap();
        assert_eq!(chat_result, "ok after retry");

        let embeddings = client.embed(&["hello".into()]).await.unwrap();
        assert_eq!(embeddings, vec![vec![0.5_f32, 0.25]]);
    }

    async fn spawn_embed_server() -> String {
        let app = Router::new().route(
            "/embeddings",
            post(|| async { Json(json!({"data": [{"embedding": [0.5, 0.25]}]})) }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn health_probe_reports_unreachable_on_503() {
        // 503 on the embeddings path (the exact wedge signature) -> probe fails.
        let app = Router::new().route(
            "/embeddings",
            post(|| async { StatusCode::SERVICE_UNAVAILABLE.into_response() }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let client = client_for(format!("http://{addr}"));
        assert!(client.check_llm().await.is_err());
    }

    #[tokio::test]
    async fn health_probe_reports_reachable_on_200() {
        let url = spawn_embed_server().await;
        let client = client_for(url);
        assert!(client.check_llm().await.is_ok());
    }

    #[tokio::test]
    async fn health_probe_is_immediate_for_mock_provider() {
        // Mock short-circuits before any network I/O, even to an unreachable host.
        let mut client = client_for("http://127.0.0.1:1".into());
        client.provider = "mock".into();
        assert!(client.check_llm().await.is_ok());
    }
}
