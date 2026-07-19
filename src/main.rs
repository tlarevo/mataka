mod api;
mod chunking;
mod engine;
mod llm;
mod store;

use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "mataka=info,tower_http=info".into()),
        )
        .init();

    let db_path = std::env::var("MATAKA_DB").unwrap_or_else(|_| "mataka.db".into());
    let port: u16 = std::env::var("MATAKA_PORT")
        .or_else(|_| std::env::var("HINDSIGHT_API_PORT"))
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8888); // same default port as Hindsight

    let state = Arc::new(api::AppState {
        store: store::Store::open(&db_path)?,
        llm: llm::LlmClient::from_env(),
    });

    tracing::info!(
        "mataka listening on :{port} | db={db_path} | llm provider={}",
        state.llm.provider
    );

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    axum::serve(listener, api::router(state)).await?;
    Ok(())
}
