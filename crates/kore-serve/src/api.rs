//! OpenAI-compatible API types and handlers with SSE streaming support.

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use kore_nn::sampler::{Rng, SamplerConfig};

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

// SSE streaming chunk types (OpenAI-compatible)
#[derive(Serialize)]
struct StreamCompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<StreamCompletionChoice>,
}

#[derive(Serialize)]
struct StreamCompletionChoice {
    text: String,
    index: usize,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct StreamChatChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<StreamChatChoice>,
}

#[derive(Serialize)]
struct StreamChatChoice {
    index: usize,
    delta: ChatDelta,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

// ============================================================================
// Helpers
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

fn make_sampler_config(temperature: f32) -> SamplerConfig {
    if temperature <= 0.0 {
        SamplerConfig::greedy()
    } else {
        SamplerConfig {
            temperature,
            ..Default::default()
        }
    }
}

// ============================================================================
// Handlers
// ============================================================================

pub async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    if req.stream {
        completions_stream(state, req).into_response()
    } else {
        completions_batch(state, req).into_response()
    }
}

fn completions_batch(state: AppState, req: CompletionRequest) -> Json<CompletionResponse> {
    let prompt_tokens = tokenize(&req.prompt);
    let prompt_len = prompt_tokens.len();
    let sampler = make_sampler_config(req.temperature);
    let mut rng = Rng::new(timestamp());

    let (response_text, gen_count) = {
        let mut guard = state.model.lock();
        if let Some(ref mut model) = *guard {
            let max_gen = req.max_tokens.min(512);
            match model.generate_with_config(&prompt_tokens, max_gen, &sampler, &mut rng) {
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

fn completions_stream(
    state: AppState,
    req: CompletionRequest,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let id = gen_id();
    let model_name = state.model_name.clone();
    let prompt_tokens = tokenize(&req.prompt);
    let prompt_len = prompt_tokens.len();
    let max_gen = req.max_tokens.min(512);
    let sampler = make_sampler_config(req.temperature);

    // Generate all tokens synchronously (mutex is not Send across yield points)
    let generated_tokens: Vec<usize> = {
        let mut rng = Rng::new(timestamp());
        let mut guard = state.model.lock();
        if let Some(ref mut model) = *guard {
            match model.generate_with_config(&prompt_tokens, max_gen, &sampler, &mut rng) {
                Ok(full_seq) => full_seq[prompt_len..].to_vec(),
                Err(_) => vec![],
            }
        } else {
            vec![]
        }
    };

    let stream = async_stream::stream! {
        for &tok in &generated_tokens {
            let text = detokenize(&[tok]);
            let chunk = StreamCompletionChunk {
                id: id.clone(),
                object: "text_completion".to_string(),
                created: timestamp(),
                model: model_name.clone(),
                choices: vec![StreamCompletionChoice {
                    text,
                    index: 0,
                    finish_reason: None,
                }],
            };
            yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
        }

        // Final chunk
        let final_chunk = StreamCompletionChunk {
            id: id.clone(),
            object: "text_completion".to_string(),
            created: timestamp(),
            model: model_name.clone(),
            choices: vec![StreamCompletionChoice {
                text: String::new(),
                index: 0,
                finish_reason: Some("stop".to_string()),
            }],
        };
        yield Ok(Event::default().data(serde_json::to_string(&final_chunk).unwrap()));
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
}

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if req.stream {
        chat_completions_stream(state, req).into_response()
    } else {
        chat_completions_batch(state, req).into_response()
    }
}

fn chat_completions_batch(state: AppState, req: ChatCompletionRequest) -> Json<ChatCompletionResponse> {
    let prompt_text: String = req.messages.iter()
        .map(|m| format!("<|{}|>{}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("");

    let prompt_tokens = tokenize(&prompt_text);
    let prompt_len = prompt_tokens.len();
    let sampler = make_sampler_config(req.temperature);
    let mut rng = Rng::new(timestamp());

    let (response_content, gen_count) = {
        let mut guard = state.model.lock();
        if let Some(ref mut model) = *guard {
            let max_gen = req.max_tokens.min(512);
            match model.generate_with_config(&prompt_tokens, max_gen, &sampler, &mut rng) {
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

// ============================================================================
// /v1/models
// ============================================================================

#[derive(Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

#[derive(Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

pub async fn list_models(State(state): State<AppState>) -> Json<ModelsResponse> {
    let model_id = if state.has_model() {
        state.model_name.clone()
    } else {
        "none".to_string()
    };

    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: model_id,
            object: "model".to_string(),
            created: timestamp(),
            owned_by: "kore".to_string(),
        }],
    })
}

fn chat_completions_stream(
    state: AppState,
    req: ChatCompletionRequest,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let id = gen_id();
    let model_name = state.model_name.clone();
    let prompt_text: String = req.messages.iter()
        .map(|m| format!("<|{}|>{}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("");
    let prompt_tokens = tokenize(&prompt_text);
    let prompt_len = prompt_tokens.len();
    let max_gen = req.max_tokens.min(512);
    let sampler = make_sampler_config(req.temperature);

    // Generate all tokens synchronously (mutex is not Send across yield points)
    let generated_tokens: Vec<usize> = {
        let mut rng = Rng::new(timestamp());
        let mut guard = state.model.lock();
        if let Some(ref mut model) = *guard {
            match model.generate_with_config(&prompt_tokens, max_gen, &sampler, &mut rng) {
                Ok(full_seq) => full_seq[prompt_len..].to_vec(),
                Err(_) => vec![],
            }
        } else {
            vec![]
        }
    };

    let stream = async_stream::stream! {
        // First chunk: role
        let role_chunk = StreamChatChunk {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: timestamp(),
            model: model_name.clone(),
            choices: vec![StreamChatChoice {
                index: 0,
                delta: ChatDelta { role: Some("assistant".to_string()), content: None },
                finish_reason: None,
            }],
        };
        yield Ok(Event::default().data(serde_json::to_string(&role_chunk).unwrap()));

        // Stream each token as a chunk
        for &tok in &generated_tokens {
            let text = detokenize(&[tok]);
            let chunk = StreamChatChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: timestamp(),
                model: model_name.clone(),
                choices: vec![StreamChatChoice {
                    index: 0,
                    delta: ChatDelta { role: None, content: Some(text) },
                    finish_reason: None,
                }],
            };
            yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
        }

        // Final chunk with finish_reason
        let final_chunk = StreamChatChunk {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: timestamp(),
            model: model_name.clone(),
            choices: vec![StreamChatChoice {
                index: 0,
                delta: ChatDelta { role: None, content: None },
                finish_reason: Some("stop".to_string()),
            }],
        };
        yield Ok(Event::default().data(serde_json::to_string(&final_chunk).unwrap()));
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
}
