// Architecture-neutral OpenAI-compatible HTTP wrapper for the verified Engine.
//
// HTTP, JSON, tokenizer, and streaming behavior are ordinary executable
// boundary code. Model execution remains behind Engine::try_add_request and
// Engine::step, with one sealed model/runtime tuple for the process lifetime.

use std::collections::HashMap;
use std::convert::Infallible;
use std::fs;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self as std_mpsc, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::{mpsc as tokio_mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;
use vosti_verus::types::BLOCK_SIZE;
use vosti_verus::exec::engine::Engine;
use vosti_verus::exec::step_observation::PlannedPrefixReuse;
use vosti_verus::exec::request_state::{AdmissionStatus, NewRequest};
use vosti_verus::boundary::tensor_runtime::{self as RT, CudaGraphOverlay};

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 8000;
const DEFAULT_MAX_MODEL_LEN: usize = 2048;
const DEFAULT_ADMISSION_QUEUE: usize = 1024;

#[cfg(test)]
#[path = "openai_server_tests.rs"]
mod tests;

#[derive(Clone)]
struct TokenizerClient {
    sender: std_mpsc::Sender<TokenizerCommand>,
}

enum TokenizerCommand {
    EncodePrompt {
        prompt: String,
        reply: oneshot::Sender<Result<Vec<u64>, String>>,
    },
    EncodeChat {
        messages_json: String,
        reply: oneshot::Sender<Result<Vec<u64>, String>>,
    },
    Decode {
        token_ids: Vec<u64>,
        reply: oneshot::Sender<Result<String, String>>,
    },
}

impl TokenizerClient {
    fn start(model_path: &str) -> Result<Self, String> {
        let (sender, receiver) = std_mpsc::channel::<TokenizerCommand>();
        let (ready_sender, ready_receiver) = std_mpsc::sync_channel(1);
        let model_path = model_path.to_string();
        thread::Builder::new()
            .name("vosti-tokenizer".to_string())
            .spawn(move || {
                let loaded = Python::with_gil(|py| -> PyResult<Py<PyAny>> {
                    let module = py.import_bound("vosti_kernels.serving_workload")?;
                    Ok(module
                        .getattr("load_serving_tokenizer")?
                        .call1((model_path,))?
                        .unbind())
                })
                .map_err(|error| format!("failed to load serving tokenizer: {error}"));
                let tokenizer = match loaded {
                    Ok(tokenizer) => {
                        let _ = ready_sender.send(Ok(()));
                        tokenizer
                    }
                    Err(error) => {
                        let _ = ready_sender.send(Err(error));
                        return;
                    }
                };

                while let Ok(command) = receiver.recv() {
                    match command {
                        TokenizerCommand::EncodePrompt { prompt, reply } => {
                            let result = Python::with_gil(|py| -> PyResult<Vec<u64>> {
                                py.import_bound("vosti_kernels.serving_workload")?
                                    .getattr("encode_serving_prompt")?
                                    .call1((tokenizer.bind(py), prompt))?
                                    .extract()
                            })
                            .map_err(|error| error.to_string());
                            let _ = reply.send(result);
                        }
                        TokenizerCommand::EncodeChat {
                            messages_json,
                            reply,
                        } => {
                            let result = Python::with_gil(|py| -> PyResult<Vec<u64>> {
                                py.import_bound("vosti_kernels.serving_workload")?
                                    .getattr("encode_serving_chat")?
                                    .call1((tokenizer.bind(py), messages_json))?
                                    .extract()
                            })
                            .map_err(|error| error.to_string());
                            let _ = reply.send(result);
                        }
                        TokenizerCommand::Decode { token_ids, reply } => {
                            let result = Python::with_gil(|py| -> PyResult<String> {
                                py.import_bound("vosti_kernels.serving_workload")?
                                    .getattr("decode_serving_tokens")?
                                    .call1((tokenizer.bind(py), token_ids))?
                                    .extract()
                            })
                            .map_err(|error| error.to_string());
                            let _ = reply.send(result);
                        }
                    }
                }
            })
            .map_err(|error| format!("failed to start tokenizer thread: {error}"))?;
        ready_receiver
            .recv()
            .map_err(|_| "tokenizer thread exited during initialization".to_string())??;
        Ok(Self { sender })
    }

    async fn encode_prompt(&self, prompt: String) -> Result<Vec<u64>, String> {
        let (reply, received) = oneshot::channel();
        self.sender
            .send(TokenizerCommand::EncodePrompt { prompt, reply })
            .map_err(|_| "tokenizer service is unavailable".to_string())?;
        received
            .await
            .map_err(|_| "tokenizer service dropped its response".to_string())?
    }

    async fn encode_chat(&self, messages_json: String) -> Result<Vec<u64>, String> {
        let (reply, received) = oneshot::channel();
        self.sender
            .send(TokenizerCommand::EncodeChat {
                messages_json,
                reply,
            })
            .map_err(|_| "tokenizer service is unavailable".to_string())?;
        received
            .await
            .map_err(|_| "tokenizer service dropped its response".to_string())?
    }

    async fn decode(&self, token_ids: Vec<u64>) -> Result<String, String> {
        let (reply, received) = oneshot::channel();
        self.sender
            .send(TokenizerCommand::Decode { token_ids, reply })
            .map_err(|_| "tokenizer service is unavailable".to_string())?;
        received
            .await
            .map_err(|_| "tokenizer service dropped its response".to_string())?
    }
}

#[derive(Debug)]
enum GenerationEvent {
    Token(u64),
    Finished {
        reason: &'static str,
        tokens: usize,
        cached_prompt_tokens: Option<usize>,
    },
    Failed(String),
    Done,
}

struct Admission {
    request_id: u64,
    prompt_tokens: Vec<u64>,
    max_tokens: usize,
    ignore_eos: bool,
    accepted: oneshot::Sender<Result<(), String>>,
    events: tokio_mpsc::UnboundedSender<GenerationEvent>,
}

enum EngineCommand {
    Admit(Admission),
    Stats(oneshot::Sender<Value>),
}

struct ActiveRequest {
    prompt_tokens: usize,
    cached_prompt_tokens: Option<usize>,
    cache_observed: bool,
    max_tokens: usize,
    emitted_tokens: usize,
    ignore_eos: bool,
    events: tokio_mpsc::UnboundedSender<GenerationEvent>,
}

// The snapshot is taken by Engine after scheduling, before forward/commit.
// Latch the first scheduled row, including cache-only chunks and completion.
fn observe_cached_prompt_tokens(
    observations: &[PlannedPrefixReuse],
    active: &mut HashMap<u64, ActiveRequest>,
) {
    for observation in observations {
        let Some(request) = active.get_mut(&observation.request_id) else {
            continue;
        };
        if !request.cache_observed {
            request.cache_observed = true;
            request.cached_prompt_tokens = observation.cached_prefix_blocks.and_then(|blocks| {
                blocks
                    .checked_mul(BLOCK_SIZE)
                    .and_then(|tokens| usize::try_from(tokens).ok())
                    .filter(|tokens| *tokens <= request.prompt_tokens)
            });
        }
    }
}

fn usage_json(
    prompt_tokens: usize,
    completion_tokens: usize,
    cached_tokens: Option<usize>,
) -> Value {
    let mut usage = json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    });
    if let Some(cached_tokens) = cached_tokens {
        usage["prompt_tokens_details"] = json!({"cached_tokens": cached_tokens});
    }
    usage
}

#[derive(Default)]
struct EngineCounters {
    accepted_requests: u64,
    completed_requests: u64,
    emitted_tokens: u64,
    engine_steps: u64,
    steps_with_multiple_live_requests: u64,
    max_live_requests: usize,
    max_waiting_requests: usize,
    max_running_requests: usize,
}

impl EngineCounters {
    fn observe_queues(&mut self, engine: &Engine) {
        self.max_live_requests = self.max_live_requests.max(engine.cs.live_requests.len());
        self.max_waiting_requests = self.max_waiting_requests.max(engine.cs.waiting.len());
        self.max_running_requests = self.max_running_requests.max(engine.cs.running.len());
    }
}

#[derive(Clone)]
struct AppState {
    commands: SyncSender<EngineCommand>,
    tokenizer: TokenizerClient,
    next_request_id: Arc<AtomicU64>,
    architecture: Arc<str>,
    model: Arc<str>,
    deployment_sha256: Arc<str>,
    max_model_len: usize,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum PromptInput {
    Text(String),
    TokenIds(Vec<u64>),
}

#[derive(Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Deserialize)]
struct CompletionRequest {
    model: Option<String>,
    prompt: PromptInput,
    max_tokens: Option<usize>,
    stream: Option<bool>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    n: Option<usize>,
    best_of: Option<usize>,
    echo: Option<bool>,
    logprobs: Option<Value>,
    stop: Option<Value>,
    #[serde(default)]
    ignore_eos: bool,
    stream_options: Option<StreamOptions>,
}

#[derive(Clone, Deserialize, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    stream: Option<bool>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    n: Option<usize>,
    stop: Option<Value>,
    #[serde(default)]
    ignore_eos: bool,
    stream_options: Option<StreamOptions>,
}

#[derive(Clone, Copy)]
enum ResponseKind {
    Completion,
    Chat,
}

struct PreparedGeneration {
    request_id: u64,
    prompt_tokens: usize,
    max_tokens: usize,
    include_usage: bool,
    events: tokio_mpsc::UnboundedReceiver<GenerationEvent>,
}

#[derive(Default)]
struct IncrementalText {
    token_ids: Vec<u64>,
    emitted_chars: usize,
}

fn is_cjk(character: char) -> bool {
    matches!(
        character as u32,
        0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2FA1F
    )
}

impl IncrementalText {
    async fn push(&mut self, tokenizer: &TokenizerClient, token_id: u64) -> Result<String, String> {
        self.token_ids.push(token_id);
        let text = tokenizer.decode(self.token_ids.clone()).await?;
        let printable_end = if text.ends_with('\n') {
            text.len()
        } else if text.chars().last().is_some_and(is_cjk) {
            text.len()
        } else {
            text.rfind(' ')
                .map_or(self.emitted_chars, |index| index + 1)
        };
        let printable = text
            .get(self.emitted_chars..printable_end)
            .unwrap_or_default()
            .to_string();
        if text.ends_with('\n') {
            self.token_ids.clear();
            self.emitted_chars = 0;
        } else {
            self.emitted_chars = printable_end;
        }
        Ok(printable)
    }

    async fn finish(&mut self, tokenizer: &TokenizerClient) -> Result<String, String> {
        let text = tokenizer.decode(self.token_ids.clone()).await?;
        let printable = text
            .get(self.emitted_chars..)
            .unwrap_or_default()
            .to_string();
        self.token_ids.clear();
        self.emitted_chars = 0;
        Ok(printable)
    }
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{name} must be a nonnegative integer"))
        })
        .unwrap_or(default)
}

fn api_error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message.into(),
                "type": "invalid_request_error"
            }
        })),
    )
        .into_response()
}

fn validate_model(requested: Option<&str>, served: &str) -> Result<(), String> {
    if let Some(requested) = requested {
        if requested != served {
            return Err(format!(
                "requested model {requested:?} does not match served model {served:?}"
            ));
        }
    }
    Ok(())
}

fn validate_greedy_options(
    temperature: Option<f64>,
    top_p: Option<f64>,
    n: Option<usize>,
    stop: Option<&Value>,
) -> Result<(), String> {
    if temperature.is_some_and(|value| value != 0.0) {
        return Err("only greedy temperature=0 is supported".to_string());
    }
    if top_p.is_some_and(|value| value != 1.0) {
        return Err("only top_p=1 is supported".to_string());
    }
    if n.is_some_and(|value| value != 1) {
        return Err("only n=1 is supported".to_string());
    }
    if stop.is_some_and(|value| !value.is_null()) {
        return Err("stop sequences are not supported".to_string());
    }
    Ok(())
}

async fn prepare_generation(
    state: &AppState,
    prompt_tokens: Vec<u64>,
    max_tokens: usize,
    ignore_eos: bool,
    include_usage: bool,
) -> Result<PreparedGeneration, Response> {
    if prompt_tokens.is_empty() {
        return Err(api_error(StatusCode::BAD_REQUEST, "prompt is empty"));
    }
    if max_tokens == 0 {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "max_tokens must be positive",
        ));
    }
    if prompt_tokens.len().saturating_add(max_tokens) > state.max_model_len {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            format!(
                "prompt plus output exceeds max model length {}",
                state.max_model_len
            ),
        ));
    }

    let request_id = state.next_request_id.fetch_add(1, Ordering::Relaxed);
    let prompt_token_count = prompt_tokens.len();
    let (accepted, acceptance) = oneshot::channel();
    let (event_sender, events) = tokio_mpsc::unbounded_channel();
    let admission = Admission {
        request_id,
        prompt_tokens,
        max_tokens,
        ignore_eos,
        accepted,
        events: event_sender,
    };
    match state.commands.try_send(EngineCommand::Admit(admission)) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => {
            return Err(api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "server admission queue is full",
            ));
        }
        Err(TrySendError::Disconnected(_)) => {
            return Err(api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "engine worker is unavailable",
            ));
        }
    }
    match acceptance.await {
        Ok(Ok(())) => Ok(PreparedGeneration {
            request_id,
            prompt_tokens: prompt_token_count,
            max_tokens,
            include_usage,
            events,
        }),
        Ok(Err(error)) => Err(api_error(StatusCode::BAD_REQUEST, error)),
        Err(_) => Err(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "engine worker dropped the admission result",
        )),
    }
}

fn chunk_json(
    kind: ResponseKind,
    id: &str,
    model: &str,
    created: u64,
    text: &str,
    token_id: Option<u64>,
    finish_reason: Option<&str>,
    usage: Option<Value>,
) -> Value {
    let choice = match kind {
        ResponseKind::Completion => json!({
            "index": 0,
            "text": text,
            "logprobs": null,
            "finish_reason": finish_reason,
            "token_id": token_id,
        }),
        ResponseKind::Chat => json!({
            "index": 0,
            "delta": {"content": text},
            "logprobs": null,
            "finish_reason": finish_reason,
            "token_id": token_id,
        }),
    };
    let object = match kind {
        ResponseKind::Completion => "text_completion.chunk",
        ResponseKind::Chat => "chat.completion.chunk",
    };
    json!({
        "id": id,
        "object": object,
        "created": created,
        "model": model,
        "choices": [choice],
        "usage": usage,
    })
}

fn streaming_response(
    state: AppState,
    kind: ResponseKind,
    prepared: PreparedGeneration,
) -> Response {
    let id = Arc::<str>::from(match kind {
        ResponseKind::Completion => format!("cmpl-{}", prepared.request_id),
        ResponseKind::Chat => format!("chatcmpl-{}", prepared.request_id),
    });
    let created = unix_seconds();
    let model = state.model.clone();
    let tokenizer = state.tokenizer.clone();
    let incremental_text = Arc::new(tokio::sync::Mutex::new(IncrementalText::default()));
    let prompt_tokens = prepared.prompt_tokens;
    let include_usage = prepared.include_usage;
    let stream = UnboundedReceiverStream::new(prepared.events).then(move |event| {
        let id = id.clone();
        let model = model.clone();
        let tokenizer = tokenizer.clone();
        let incremental_text = incremental_text.clone();
        async move {
            let data = match event {
                GenerationEvent::Token(token_id) => {
                    match incremental_text
                        .lock()
                        .await
                        .push(&tokenizer, token_id)
                        .await
                    {
                        Ok(text) => chunk_json(
                            kind,
                            &id,
                            &model,
                            created,
                            &text,
                            Some(token_id),
                            None,
                            None,
                        )
                        .to_string(),
                        Err(error) => json!({
                            "error": {"message": error, "type": "server_error"}
                        })
                        .to_string(),
                    }
                }
                GenerationEvent::Finished {
                    reason,
                    tokens,
                    cached_prompt_tokens,
                } => {
                    let usage = include_usage
                        .then(|| usage_json(prompt_tokens, tokens, cached_prompt_tokens));
                    match incremental_text.lock().await.finish(&tokenizer).await {
                        Ok(text) => {
                            chunk_json(kind, &id, &model, created, &text, None, Some(reason), usage)
                                .to_string()
                        }
                        Err(error) => json!({
                            "error": {"message": error, "type": "server_error"}
                        })
                        .to_string(),
                    }
                }
                GenerationEvent::Failed(error) => json!({
                    "error": {"message": error, "type": "server_error"}
                })
                .to_string(),
                GenerationEvent::Done => "[DONE]".to_string(),
            };
            Ok::<Event, Infallible>(Event::default().data(data))
        }
    });
    Sse::new(stream)
        .keep_alive(KeepAlive::new())
        .into_response()
}

async fn nonstreaming_response(
    state: AppState,
    kind: ResponseKind,
    mut prepared: PreparedGeneration,
) -> Response {
    let mut token_ids = Vec::with_capacity(prepared.max_tokens);
    let mut finish_reason = "length";
    let mut cached_prompt_tokens = None;
    while let Some(event) = prepared.events.recv().await {
        match event {
            GenerationEvent::Token(token_id) => token_ids.push(token_id),
            GenerationEvent::Finished {
                reason,
                cached_prompt_tokens: cached,
                ..
            } => {
                finish_reason = reason;
                cached_prompt_tokens = cached;
            }
            GenerationEvent::Failed(error) => {
                return api_error(StatusCode::INTERNAL_SERVER_ERROR, error);
            }
            GenerationEvent::Done => break,
        }
    }
    let text = match state.tokenizer.decode(token_ids.clone()).await {
        Ok(text) => text,
        Err(error) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, error),
    };
    let object = match kind {
        ResponseKind::Completion => "text_completion",
        ResponseKind::Chat => "chat.completion",
    };
    let choice = match kind {
        ResponseKind::Completion => json!({
            "index": 0,
            "text": text,
            "logprobs": null,
            "finish_reason": finish_reason,
        }),
        ResponseKind::Chat => json!({
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "logprobs": null,
            "finish_reason": finish_reason,
        }),
    };
    Json(json!({
        "id": match kind {
            ResponseKind::Completion => format!("cmpl-{}", prepared.request_id),
            ResponseKind::Chat => format!("chatcmpl-{}", prepared.request_id),
        },
        "object": object,
        "created": unix_seconds(),
        "model": state.model.as_ref(),
        "choices": [choice],
        "usage": usage_json(prepared.prompt_tokens, token_ids.len(), cached_prompt_tokens),
    }))
    .into_response()
}

async fn completions(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> Response {
    if let Err(error) = validate_model(request.model.as_deref(), &state.model) {
        return api_error(StatusCode::NOT_FOUND, error);
    }
    if let Err(error) = validate_greedy_options(
        request.temperature,
        request.top_p,
        request.n,
        request.stop.as_ref(),
    ) {
        return api_error(StatusCode::BAD_REQUEST, error);
    }
    if request.best_of.is_some_and(|value| value != 1)
        || request.echo.unwrap_or(false)
        || request.logprobs.is_some()
    {
        return api_error(
            StatusCode::BAD_REQUEST,
            "best_of, echo, and logprobs are unsupported",
        );
    }
    let prompt_tokens = match request.prompt {
        PromptInput::Text(prompt) => match state.tokenizer.encode_prompt(prompt).await {
            Ok(tokens) => tokens,
            Err(error) => return api_error(StatusCode::BAD_REQUEST, error),
        },
        PromptInput::TokenIds(tokens) => tokens,
    };
    let include_usage = request
        .stream_options
        .is_some_and(|options| options.include_usage);
    let prepared = match prepare_generation(
        &state,
        prompt_tokens,
        request.max_tokens.unwrap_or(16),
        request.ignore_eos,
        include_usage,
    )
    .await
    {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };
    if request.stream.unwrap_or(false) {
        streaming_response(state, ResponseKind::Completion, prepared)
    } else {
        nonstreaming_response(state, ResponseKind::Completion, prepared).await
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if let Err(error) = validate_model(request.model.as_deref(), &state.model) {
        return api_error(StatusCode::NOT_FOUND, error);
    }
    if let Err(error) = validate_greedy_options(
        request.temperature,
        request.top_p,
        request.n,
        request.stop.as_ref(),
    ) {
        return api_error(StatusCode::BAD_REQUEST, error);
    }
    let max_tokens = match (request.max_tokens, request.max_completion_tokens) {
        (Some(_), Some(_)) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "specify only one of max_tokens and max_completion_tokens",
            );
        }
        (Some(value), None) | (None, Some(value)) => value,
        (None, None) => 16,
    };
    let messages_json = match serde_json::to_string(&request.messages) {
        Ok(messages) => messages,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, error.to_string()),
    };
    let prompt_tokens = match state.tokenizer.encode_chat(messages_json).await {
        Ok(tokens) => tokens,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, error),
    };
    let include_usage = request
        .stream_options
        .is_some_and(|options| options.include_usage);
    let prepared = match prepare_generation(
        &state,
        prompt_tokens,
        max_tokens,
        request.ignore_eos,
        include_usage,
    )
    .await
    {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };
    if request.stream.unwrap_or(false) {
        streaming_response(state, ResponseKind::Chat, prepared)
    } else {
        nonstreaming_response(state, ResponseKind::Chat, prepared).await
    }
}

async fn request_engine_stats(state: &AppState) -> Result<Value, Response> {
    let (reply, received) = oneshot::channel();
    match state.commands.try_send(EngineCommand::Stats(reply)) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => {
            return Err(api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "engine queue is full",
            ));
        }
        Err(TrySendError::Disconnected(_)) => {
            return Err(api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "engine worker is unavailable",
            ));
        }
    }
    received.await.map_err(|_| {
        api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "engine worker dropped the stats response",
        )
    })
}

async fn health(State(state): State<AppState>) -> Response {
    match request_engine_stats(&state).await {
        Ok(_) => Json(json!({"status": "ok"})).into_response(),
        Err(response) => response,
    }
}

async fn models(State(state): State<AppState>) -> impl IntoResponse {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model.as_ref(),
            "object": "model",
            "created": 0,
            "owned_by": "vosti-verus",
        }]
    }))
}

async fn stats(State(state): State<AppState>) -> Response {
    match request_engine_stats(&state).await {
        Ok(engine) => Json(json!({
            "server": {
                "architecture": state.architecture.as_ref(),
                "model": state.model.as_ref(),
                "deployment_sha256": state.deployment_sha256.as_ref(),
            },
            "engine": engine,
        }))
        .into_response(),
        Err(response) => response,
    }
}

fn process_command(
    command: EngineCommand,
    engine: &mut Engine,
    overlay: Option<&CudaGraphOverlay>,
    eos_token_ids: &[u64],
    active: &mut HashMap<u64, ActiveRequest>,
    counters: &mut EngineCounters,
) {
    match command {
        EngineCommand::Admit(admission) => {
            let prompt_tokens = admission.prompt_tokens.len();
            let status = engine.try_add_request(NewRequest {
                request_id: admission.request_id,
                prompt_tokens: admission.prompt_tokens,
                max_tokens: admission.max_tokens,
                eos_token_ids: eos_token_ids.to_vec(),
                ignore_eos: admission.ignore_eos,
            });
            if status == AdmissionStatus::Accepted {
                active.insert(
                    admission.request_id,
                    ActiveRequest {
                        prompt_tokens,
                        cached_prompt_tokens: None,
                        cache_observed: false,
                        max_tokens: admission.max_tokens,
                        emitted_tokens: 0,
                        ignore_eos: admission.ignore_eos,
                        events: admission.events,
                    },
                );
                counters.accepted_requests += 1;
                counters.observe_queues(engine);
                let _ = admission.accepted.send(Ok(()));
            } else {
                let error = format!("engine rejected request: {status:?}");
                let _ = admission
                    .events
                    .send(GenerationEvent::Failed(error.clone()));
                let _ = admission.events.send(GenerationEvent::Done);
                let _ = admission.accepted.send(Err(error));
            }
        }
        EngineCommand::Stats(reply) => {
            let graph = overlay
                .map(RT::cuda_graph_overlay_stats_json)
                .and_then(|value| serde_json::from_str::<Value>(&value).ok());
            let _ = reply.send(json!({
                "accepted_requests": counters.accepted_requests,
                "completed_requests": counters.completed_requests,
                "emitted_tokens": counters.emitted_tokens,
                "engine_steps": counters.engine_steps,
                "steps_with_multiple_live_requests": counters.steps_with_multiple_live_requests,
                "max_live_requests": counters.max_live_requests,
                "max_waiting_requests": counters.max_waiting_requests,
                "max_running_requests": counters.max_running_requests,
                "max_num_seqs": engine.cs.config.max_num_seqs,
                "max_num_batched_tokens": engine.cs.config.max_num_batched_tokens,
                "num_kv_blocks": engine.cs.num_blocks,
                "free_kv_blocks": engine.cs.free_blocks,
                "live_requests": engine.cs.live_requests.len(),
                "waiting_requests": engine.cs.waiting.len(),
                "running_requests": engine.cs.running.len(),
                "cuda_graph": graph,
            }));
        }
    }
}

fn run_engine_loop(
    mut engine: Engine,
    overlay: Option<CudaGraphOverlay>,
    commands: Receiver<EngineCommand>,
    eos_token_ids: Vec<u64>,
) {
    let mut active = HashMap::<u64, ActiveRequest>::new();
    let mut counters = EngineCounters::default();

    loop {
        if engine.cs.live_requests.is_empty() {
            let command = match commands.recv() {
                Ok(command) => command,
                Err(_) => return,
            };
            process_command(
                command,
                &mut engine,
                overlay.as_ref(),
                &eos_token_ids,
                &mut active,
                &mut counters,
            );
        }
        loop {
            match commands.try_recv() {
                Ok(command) => process_command(
                    command,
                    &mut engine,
                    overlay.as_ref(),
                    &eos_token_ids,
                    &mut active,
                    &mut counters,
                ),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        if engine.cs.live_requests.is_empty() {
            continue;
        }

        counters.engine_steps += 1;
        if engine.cs.live_requests.len() > 1 {
            counters.steps_with_multiple_live_requests += 1;
        }
        let (emitted, _samples, _reprs) = engine.step(overlay.as_ref());
        observe_cached_prompt_tokens(&engine.last_step_prefix_reuse, &mut active);
        counters.observe_queues(&engine);
        let request_ids = active.keys().copied().collect::<Vec<_>>();
        for request_id in request_ids {
            let Some(request) = active.get_mut(&request_id) else {
                continue;
            };
            if let Some(token) = emitted.get(&request_id) {
                request.emitted_tokens += 1;
                counters.emitted_tokens += 1;
                let _ = request.events.send(GenerationEvent::Token(*token));
            }
            if !engine.cs.live_requests.contains_key(&request_id) {
                let reason = if !request.ignore_eos
                    && emitted
                        .get(&request_id)
                        .is_some_and(|token| eos_token_ids.contains(token))
                    && request.emitted_tokens < request.max_tokens
                {
                    "stop"
                } else {
                    "length"
                };
                let _ = request.events.send(GenerationEvent::Finished {
                    reason,
                    tokens: request.emitted_tokens,
                    cached_prompt_tokens: request.cached_prompt_tokens,
                });
                let _ = request.events.send(GenerationEvent::Done);
                active.remove(&request_id);
                counters.completed_requests += 1;
            }
        }
    }
}

pub fn run_openai_server(
    architecture: &str,
    model_path: &str,
    deployment_bundle: &str,
    eos_token_ids: Vec<u64>,
    engine: Engine,
    overlay: Option<CudaGraphOverlay>,
) -> Result<(), String> {
    let model = std::env::var("VOSTI_SERVED_MODEL_NAME").unwrap_or_else(|_| {
        Path::new(model_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(architecture)
            .to_string()
    });
    let host = std::env::var("VOSTI_SERVER_HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port = std::env::var("VOSTI_SERVER_PORT")
        .ok()
        .map(|value| {
            value
                .parse::<u16>()
                .unwrap_or_else(|_| panic!("VOSTI_SERVER_PORT must be a valid port"))
        })
        .unwrap_or(DEFAULT_PORT);
    let address = format!("{host}:{port}")
        .parse::<SocketAddr>()
        .map_err(|error| format!("invalid server address {host}:{port}: {error}"))?;
    let max_model_len = env_usize("VOSTI_MAX_MODEL_LEN", DEFAULT_MAX_MODEL_LEN);
    let queue_capacity = env_usize("VOSTI_ADMISSION_QUEUE", DEFAULT_ADMISSION_QUEUE);
    if max_model_len == 0 || queue_capacity == 0 {
        return Err("server length and queue limits must be positive".to_string());
    }
    let deployment_path = Path::new(deployment_bundle).join("deployment.json");
    let deployment: Value = serde_json::from_str(
        &fs::read_to_string(&deployment_path)
            .map_err(|error| format!("failed to read {}: {error}", deployment_path.display()))?,
    )
    .map_err(|error| format!("invalid {}: {error}", deployment_path.display()))?;
    let deployment_sha256 = deployment
        .get("deployment_sha256")
        .and_then(Value::as_str)
        .filter(|value| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
        .ok_or_else(|| {
            format!(
                "{} has no lowercase SHA-256 deployment identity",
                deployment_path.display()
            )
        })?;

    let tokenizer = TokenizerClient::start(model_path)?;
    let (command_sender, command_receiver) = std_mpsc::sync_channel(queue_capacity);
    let state = AppState {
        commands: command_sender,
        tokenizer,
        next_request_id: Arc::new(AtomicU64::new(0)),
        architecture: Arc::from(architecture),
        model: Arc::from(model.clone()),
        deployment_sha256: Arc::from(deployment_sha256),
        max_model_len,
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(stats))
        .route("/v1/models", get(models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);
    let (ready_sender, ready_receiver) = std_mpsc::sync_channel(1);
    thread::Builder::new()
        .name("vosti-http".to_string())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("failed to build HTTP runtime");
            runtime.block_on(async move {
                match tokio::net::TcpListener::bind(address).await {
                    Ok(listener) => {
                        let _ = ready_sender.send(Ok(()));
                        if let Err(error) = axum::serve(listener, app).await {
                            eprintln!("OpenAI server failed: {error}");
                        }
                    }
                    Err(error) => {
                        let _ = ready_sender.send(Err(format!(
                            "failed to bind OpenAI server at {address}: {error}"
                        )));
                    }
                }
            });
        })
        .map_err(|error| format!("failed to start HTTP thread: {error}"))?;
    ready_receiver
        .recv()
        .map_err(|_| "HTTP thread exited during initialization".to_string())??;
    println!("SERVER_READY architecture={architecture} model={model} address=http://{address}");
    run_engine_loop(engine, overlay, command_receiver, eos_token_ids);
    Ok(())
}
