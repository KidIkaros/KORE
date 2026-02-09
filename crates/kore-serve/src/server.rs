//! Server setup and configuration.

use axum::{routing::{get, post}, Router};
use tower_http::cors::CorsLayer;

use crate::api;
use crate::health;
use crate::state::AppState;

/// Build the Kore inference server router with shared state.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health::health))
        .route("/v1/completions", post(api::completions))
        .route("/v1/chat/completions", post(api::chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Start the server on the given address with no model loaded (placeholder mode).
pub async fn serve(addr: &str) -> anyhow::Result<()> {
    serve_with_state(addr, AppState::empty()).await
}

/// Start the server on the given address with the given state.
pub async fn serve_with_state(addr: &str, state: AppState) -> anyhow::Result<()> {
    let app = build_router(state);

    tracing::info!("Kore inference server starting on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
