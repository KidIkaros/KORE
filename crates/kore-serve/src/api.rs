//! OpenAI-compatible API types and handlers.

use axum::Json;
use serde::{Deserialize, Serialize};

// ============================================================================
// Request types
// ============================================================================

#[derive(Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> usize { 128 }
fn default_temperature() -> f32 { 1.0 }

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

// ============================================================================
// Response types
// ============================================================================

#[derive(Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// ============================================================================
// Handlers (placeholder — real inference requires model loading)
// ============================================================================

fn timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn gen_id() -> String {
    format!("kore-{:x}", timestamp())
}

pub async fn completions(Json(req): Json<CompletionRequest>) -> Json<CompletionResponse> {
    // Placeholder: echo back the prompt with a note
    let response_text = format!(
        "[Kore inference placeholder] Model '{}' received prompt of {} chars. \
         Real inference requires a loaded model.",
        req.model,
        req.prompt.len()
    );

    Json(CompletionResponse {
        id: gen_id(),
        object: "text_completion".to_string(),
        created: timestamp(),
        model: req.model,
        choices: vec![CompletionChoice {
            text: response_text,
            index: 0,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: req.prompt.split_whitespace().count(),
            completion_tokens: 0,
            total_tokens: req.prompt.split_whitespace().count(),
        },
    })
}

pub async fn chat_completions(Json(req): Json<ChatCompletionRequest>) -> Json<ChatCompletionResponse> {
    let prompt_tokens: usize = req.messages.iter()
        .map(|m| m.content.split_whitespace().count())
        .sum();

    let response_content = format!(
        "[Kore inference placeholder] Model '{}' received {} messages. \
         Real inference requires a loaded model.",
        req.model,
        req.messages.len()
    );

    Json(ChatCompletionResponse {
        id: gen_id(),
        object: "chat.completion".to_string(),
        created: timestamp(),
        model: req.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: response_content,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: 0,
            total_tokens: prompt_tokens,
        },
    })
}
