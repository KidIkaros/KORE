//! Server setup and configuration.

use axum::{routing::{get, post}, Router};
use tower_http::cors::CorsLayer;

use crate::api;
use crate::health;

/// Build the Kore inference server router.
pub fn build_router() -> Router {
    Router::new()
        .route("/health", get(health::health))
        .route("/v1/completions", post(api::completions))
        .route("/v1/chat/completions", post(api::chat_completions))
        .layer(CorsLayer::permissive())
}

/// Start the server on the given address.
pub async fn serve(addr: &str) -> anyhow::Result<()> {
    let app = build_router();

    tracing::info!("Kore inference server starting on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
