use anyhow::{Context, Result};
use axum::{
    extract::{State, WebSocketUpgrade},
    http::header,
    response::IntoResponse,
    routing::get,
    Router,
};
use config::AppState;
use std::{env, net::SocketAddr};
use tracing::{info, warn};

mod audio;
mod config;
mod humanize;
mod openai_realtime;
mod response_policy;
mod signals;
mod split_providers;
mod tracking;
mod twilio;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    let env_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".env");

    match dotenvy::from_path_override(&env_path) {
        Ok(_) => info!("loaded .env from {:?}", env_path),
        Err(e) => warn!("failed to load .env from {:?}: {}", env_path, e),
    }

    info!("current working directory: {:?}", std::env::current_dir()?);
    info!("PUBLIC_WS_URL env value: {:?}", env::var("PUBLIC_WS_URL"));

    let state = AppState::from_env()?;
    info!("orchestrator stack: {:?}", state.orchestrator_stack);

    let app = Router::new()
        .route("/health", get(health))
        .route("/twiml", get(twiml))
        .route("/twilio/media", get(twilio_ws))
        .with_state(state);

    let addr: SocketAddr = env::var("BIND_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:3000".to_string())
        .parse()
        .context("invalid BIND_ADDR")?;

    info!("listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health() -> impl IntoResponse {
    "ok"
}

async fn twiml(State(state): State<AppState>) -> impl IntoResponse {
    let response = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
            <Response>
            <Connect>
                <Stream url="{}" />
            </Connect>
            </Response>"#,
        state.public_ws_url
    );

    ([(header::CONTENT_TYPE, "text/xml")], response)
}

async fn twilio_ws(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| async move {
        if let Err(e) = twilio::handle_twilio_socket(socket, state).await {
            warn!("twilio socket error: {:?}", e);
        }
    })
}
