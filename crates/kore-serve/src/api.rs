//! OpenAI-compatible API types and handlers.

use axum::extract::State;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::state::AppState;

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
// Handlers
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

/// Simple byte-level tokenizer: each byte is a token ID.
fn tokenize(text: &str) -> Vec<usize> {
    text.bytes().map(|b| b as usize).collect()
}

/// Simple byte-level detokenizer.
fn detokenize(ids: &[usize]) -> String {
    ids.iter()
        .map(|&id| if id < 128 { id as u8 as char } else { '?' })
        .collect()
}

pub async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Json<CompletionResponse> {
    let prompt_tokens = tokenize(&req.prompt);
    let prompt_len = prompt_tokens.len();

    let (response_text, gen_count) = {
        let mut guard = state.model.lock();
        if let Some(ref mut model) = *guard {
            let max_gen = req.max_tokens.min(512);
            match model.generate(&prompt_tokens, max_gen) {
                Ok(full_seq) => {
                    let generated = &full_seq[prompt_len..];
                    (detokenize(generated), generated.len())
                }
                Err(e) => (format!("[error: {}]", e), 0),
            }
        } else {
            (format!(
                "[Kore] No model loaded. Model '{}', prompt {} tokens.",
                req.model, prompt_len
            ), 0)
        }
    };

    Json(CompletionResponse {
        id: gen_id(),
        object: "text_completion".to_string(),
        created: timestamp(),
        model: state.model_name.clone(),
        choices: vec![CompletionChoice {
            text: response_text,
            index: 0,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens: gen_count,
            total_tokens: prompt_len + gen_count,
        },
    })
}

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Json<ChatCompletionResponse> {
    // Concatenate messages into a single prompt
    let prompt_text: String = req.messages.iter()
        .map(|m| format!("<|{}|>{}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("");

    let prompt_tokens = tokenize(&prompt_text);
    let prompt_len = prompt_tokens.len();

    let (response_content, gen_count) = {
        let mut guard = state.model.lock();
        if let Some(ref mut model) = *guard {
            let max_gen = req.max_tokens.min(512);
            match model.generate(&prompt_tokens, max_gen) {
                Ok(full_seq) => {
                    let generated = &full_seq[prompt_len..];
                    (detokenize(generated), generated.len())
                }
                Err(e) => (format!("[error: {}]", e), 0),
            }
        } else {
            (format!(
                "[Kore] No model loaded. Model '{}', {} messages.",
                req.model, req.messages.len()
            ), 0)
        }
    };

    Json(ChatCompletionResponse {
        id: gen_id(),
        object: "chat.completion".to_string(),
        created: timestamp(),
        model: state.model_name.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: response_content,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens: gen_count,
            total_tokens: prompt_len + gen_count,
        },
    })
}
