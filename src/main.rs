mod api;
mod chunking;
mod engine;
mod llm;
mod store;
mod tokens;
mod vet;

use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // CLI subcommand dispatch
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("vet") {
        return run_vet_subcommand(&args[2..]).await;
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "mataka=info,tower_http=info".into()),
        )
        .init();

    let db_path = std::env::var("MATAKA_DB").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        format!("{}/.local/share/mataka", home)
    });
    let port: u16 = std::env::var("MATAKA_PORT")
        .or_else(|_| std::env::var("HINDSIGHT_API_PORT"))
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8888);

    // Detect legacy single-file mode: if path exists and is a file (not a directory)
    let path = std::path::Path::new(&db_path);
    let store = if path.exists() && path.is_file() {
        tracing::warn!(
            "DEPRECATED: single-file MATAKA_DB={db_path} is legacy mode. \
             Migrate to a directory path (e.g. ./mataka-data/) for per-bank sharding."
        );
        store::Store::open_legacy(&db_path)?
    } else {
        store::Store::open(path)?
    };

    let state = Arc::new(api::AppState {
        store,
        llm: llm::LlmClient::from_env(),
    });

    tracing::info!(
        "mataka listening on :{port} | db={db_path} | llm provider={}",
        state.llm.provider
    );

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;

    // Graceful shutdown on SIGTERM (brew services) and SIGINT (Ctrl+C)
    let shutdown_signal = async {
        let ctrl_c = tokio::signal::ctrl_c();
        #[cfg(unix)]
        let sigterm = async {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("failed to install SIGTERM handler")
                .recv()
                .await;
        };
        #[cfg(not(unix))]
        let sigterm = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => tracing::info!("received SIGINT, shutting down"),
            _ = sigterm => tracing::info!("received SIGTERM, shutting down"),
        }
    };

    axum::serve(listener, api::router(state))
        .with_graceful_shutdown(shutdown_signal)
        .await?;
    Ok(())
}

/// `mataka vet [--bank <id>] [--fix]` — scan existing data for secrets.
async fn run_vet_subcommand(args: &[String]) -> anyhow::Result<()> {
    let mut bank_id: Option<String> = None;
    let mut fix = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--bank" => {
                i += 1;
                bank_id = args.get(i).cloned();
            }
            "--fix" => fix = true,
            "--help" | "-h" => {
                eprintln!("Usage: mataka vet [--bank <id>] [--fix]");
                eprintln!("  --bank <id>  Scan a specific bank (default: all banks)");
                eprintln!("  --fix        Apply redaction to detected secrets");
                return Ok(());
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let db_path = std::env::var("MATAKA_DB").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        format!("{}/.local/share/mataka", home)
    });
    let path = std::path::Path::new(&db_path);
    let store = if path.exists() && path.is_file() {
        store::Store::open_legacy(&db_path)?
    } else {
        store::Store::open(path)?
    };

    let engine = vet::VetEngine::new()?;
    let llm = llm::LlmClient::from_env();
    let banks = if let Some(ref bid) = bank_id {
        vec![bid.clone()]
    } else {
        store.list_bank_ids()?
    };

    let mut total_findings = 0usize;
    let mut total_fixed = 0u64;

    for bid in &banks {
        let items = store.scan_units_for_vet(bid)?;
        for item in &items {
            let result = engine.scan(&item.value);
            if result.detections.is_empty() {
                continue;
            }
            total_findings += result.detections.len();
            for det in &result.detections {
                println!(
                    "[{bid}] {} ({}) {}x in {}.{} id={}",
                    det.rule_id, item.source, det.count, item.source, item.field, item.id
                );
            }
            if fix && item.field == "text" && item.source == "unit" {
                store.quarantine_unit(bid, &item.id)?;
                let (redacted, _) = engine.vet_text(&item.value, vet::VetMode::Redact)?;
                let emb = llm.embed(std::slice::from_ref(&redacted)).await?;
                store.redact_unit(bid, &item.id, &redacted, &emb[0])?;
                total_fixed += 1;
                println!("[{bid}] Fixed unit {}", item.id);
            }
        }
    }

    println!("\nScan complete: {total_findings} findings, {total_fixed} fixed");
    Ok(())
}
