use crate::audio::{pcm_i16_to_ulaw, pcm_s16le_to_i16, TWILIO_ULAW_FRAME_BYTES, ULAW_BYTES_PER_MS};
use crate::config::AppState;
use crate::humanize::{humanization_metrics, AudioHumanizer, HumanizedSpeech};
use crate::response_policy::{response_budget, response_instructions_for_budget};
use crate::signals::RealtimeSignals;
use crate::tracking::CallTracker;
use anyhow::{anyhow, Context, Result};
use base64::{engine::general_purpose, Engine as _};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex};
use tokio_tungstenite::{
    connect_async,
    tungstenite::{client::IntoClientRequest, http::HeaderValue, Message as TungsteniteMessage},
};
use tracing::{debug, error, info, warn};

const CARTESIA_VERSION: &str = "2026-03-01";
const PREDICTIVE_TRIGGER_MS: u64 = 120;

pub(crate) struct SplitProviderHandle {
    pub(crate) audio_tx: mpsc::UnboundedSender<Vec<u8>>,
    pub(crate) control_tx: mpsc::UnboundedSender<SplitControl>,
}

#[derive(Clone, Debug)]
struct UserUtterance {
    user_generation: u64,
    text: String,
    predictive: bool,
    received_at: Instant,
}

#[derive(Clone, Debug)]
pub(crate) enum SplitControl {
    Speak { generation: u64, text: String },
    Cancel { context_id: Option<String> },
}

pub(crate) async fn connect_split_providers(
    state: AppState,
    stream_sid: String,
    twilio_out_tx: mpsc::Sender<Value>,
    signals: RealtimeSignals,
    call_tracker: Option<CallTracker>,
) -> Result<SplitProviderHandle> {
    let deepgram_api_key = state
        .deepgram_api_key
        .clone()
        .context("DEEPGRAM_API_KEY missing for ORCHESTRATOR_STACK=split")?;
    let cartesia_api_key = state
        .cartesia_api_key
        .clone()
        .context("CARTESIA_API_KEY missing for ORCHESTRATOR_STACK=split")?;
    state
        .cartesia_voice_id
        .clone()
        .context("CARTESIA_VOICE_ID missing for ORCHESTRATOR_STACK=split")?;
    info!(
        "humanization config: {}",
        humanization_metrics(&state.humanization)
    );

    let (deepgram_audio_tx, deepgram_audio_rx) = mpsc::unbounded_channel::<Vec<u8>>();
    let (llm_tx, llm_rx) = mpsc::unbounded_channel::<UserUtterance>();
    let (cartesia_tx, cartesia_rx) = mpsc::unbounded_channel::<SplitControl>();

    spawn_deepgram_stream(
        state.clone(),
        deepgram_api_key,
        deepgram_audio_rx,
        llm_tx,
        call_tracker.clone(),
    );
    spawn_openai_text_worker(
        state.clone(),
        signals.clone(),
        llm_rx,
        cartesia_tx.clone(),
        call_tracker,
    );
    spawn_cartesia_tts(
        state,
        cartesia_api_key,
        stream_sid,
        twilio_out_tx,
        signals,
        cartesia_rx,
    );

    Ok(SplitProviderHandle {
        audio_tx: deepgram_audio_tx,
        control_tx: cartesia_tx,
    })
}

fn spawn_deepgram_stream(
    state: AppState,
    api_key: String,
    mut audio_rx: mpsc::UnboundedReceiver<Vec<u8>>,
    llm_tx: mpsc::UnboundedSender<UserUtterance>,
    call_tracker: Option<CallTracker>,
) {
    tokio::spawn(async move {
        let url = format!(
            "wss://api.deepgram.com/v1/listen?model={}&language={}&encoding=mulaw&sample_rate=8000&channels=1&interim_results=true&endpointing={}&vad_events=true&smart_format=true",
            state.deepgram_model,
            state.deepgram_language,
            state.deepgram_endpointing_ms
        );
        let mut request = match url.into_client_request() {
            Ok(request) => request,
            Err(e) => {
                error!("failed building Deepgram request: {}", e);
                return;
            }
        };
        let auth_value = format!("Token {}", api_key);
        if let Ok(header_value) = HeaderValue::from_str(&auth_value) {
            request.headers_mut().insert("Authorization", header_value);
        }

        info!("connecting to Deepgram streaming STT");
        let (ws, _) = match connect_async(request).await {
            Ok(connected) => connected,
            Err(e) => {
                error!("failed connecting to Deepgram: {}", e);
                return;
            }
        };
        info!("connected to Deepgram streaming STT");

        let (mut dg_sender, mut dg_receiver) = ws.split();

        tokio::spawn(async move {
            while let Some(bytes) = audio_rx.recv().await {
                if let Err(e) = dg_sender.send(TungsteniteMessage::Binary(bytes)).await {
                    warn!("failed sending audio to Deepgram: {}", e);
                    break;
                }
            }
        });

        let mut final_parts: Vec<String> = Vec::new();
        let mut user_generation: u64 = 1;
        let predictive_seq = Arc::new(AtomicU64::new(0));
        let predicted_generation = Arc::new(AtomicU64::new(0));

        while let Some(msg) = dg_receiver.next().await {
            let msg = match msg {
                Ok(msg) => msg,
                Err(e) => {
                    warn!("Deepgram receive error: {}", e);
                    break;
                }
            };

            let text = match msg {
                TungsteniteMessage::Text(text) => text,
                TungsteniteMessage::Close(_) => {
                    info!("Deepgram websocket closed");
                    break;
                }
                _ => continue,
            };

            let value: Value = match serde_json::from_str(&text) {
                Ok(value) => value,
                Err(e) => {
                    warn!("bad Deepgram JSON: {}", e);
                    continue;
                }
            };

            if value.get("type").and_then(|v| v.as_str()) == Some("SpeechStarted") {
                user_generation += 1;
                final_parts.clear();
                predictive_seq.fetch_add(1, Ordering::Relaxed);
                continue;
            }

            let transcript = deepgram_transcript(&value).trim().to_string();
            let is_final = value
                .get("is_final")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let speech_final = value
                .get("speech_final")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if !is_final && !transcript.is_empty() && predictive_candidate(&transcript) {
                let seq = predictive_seq.fetch_add(1, Ordering::Relaxed) + 1;
                let llm_tx_for_task = llm_tx.clone();
                let text_for_task = transcript.clone();
                let predicted_generation_for_task = predicted_generation.clone();
                let predictive_seq_for_task = predictive_seq.clone();

                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_millis(PREDICTIVE_TRIGGER_MS)).await;
                    if predictive_seq_for_task.load(Ordering::Relaxed) != seq {
                        return;
                    }
                    if predicted_generation_for_task.swap(user_generation, Ordering::Relaxed)
                        == user_generation
                    {
                        return;
                    }

                    info!(
                        "deepgram predictive utterance generation={} text={:?}",
                        user_generation, text_for_task
                    );
                    let _ = llm_tx_for_task.send(UserUtterance {
                        user_generation,
                        text: text_for_task,
                        predictive: true,
                        received_at: Instant::now(),
                    });
                });
            }

            if is_final && !transcript.is_empty() {
                final_parts.push(transcript);
            }

            if speech_final {
                let utterance = final_parts.join(" ").trim().to_string();
                final_parts.clear();

                if utterance.is_empty() {
                    continue;
                }

                info!(
                    "deepgram final utterance generation={} text={:?}",
                    user_generation, utterance
                );
                if let Some(tracker) = call_tracker.as_ref() {
                    tracker.record_user_text(utterance.clone());
                }
                if let Err(e) = llm_tx.send(UserUtterance {
                    user_generation,
                    text: utterance,
                    predictive: false,
                    received_at: Instant::now(),
                }) {
                    warn!("failed sending transcript to LLM worker: {}", e);
                    break;
                }
            }
        }
    });
}

fn spawn_openai_text_worker(
    state: AppState,
    signals: RealtimeSignals,
    mut llm_rx: mpsc::UnboundedReceiver<UserUtterance>,
    cartesia_tx: mpsc::UnboundedSender<SplitControl>,
    call_tracker: Option<CallTracker>,
) {
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        let history = Arc::new(Mutex::new(Vec::<(String, String)>::new()));
        let mut answered_user_generation = 0_u64;

        while let Some(mut utterance) = llm_rx.recv().await {
            while let Ok(newer_utterance) = llm_rx.try_recv() {
                info!("dropping older pending utterance; answering latest");
                utterance = newer_utterance;
            }

            if utterance.user_generation <= answered_user_generation {
                debug!(
                    "skipping already answered user generation={} predictive={}",
                    utterance.user_generation, utterance.predictive
                );
                continue;
            }
            answered_user_generation = utterance.user_generation;

            let generation = signals.response_generation.fetch_add(1, Ordering::Relaxed) + 1;
            info!(
                "latency split_transcript_to_llm_start={}ms generation={} predictive={}",
                utterance.received_at.elapsed().as_millis(),
                generation,
                utterance.predictive
            );
            signals.generated_audio_ms.store(0, Ordering::Relaxed);
            signals.played_audio_ms.store(0, Ordering::Relaxed);
            signals.queued_audio_ms.store(0, Ordering::Relaxed);
            signals.pending_playback_marks.store(0, Ordering::Relaxed);
            signals.response_active.store(true, Ordering::Relaxed);

            {
                let mut history = history.lock().await;
                history.push(("user".to_string(), utterance.text.clone()));
                keep_recent_history(&mut history, 10);
            }

            let response =
                match stream_openai_text_response(&client, &state, history.clone(), generation)
                    .await
                {
                    Ok(response) => response,
                    Err(e) => {
                        error!("OpenAI text error: {:?}", e);
                        signals.response_active.store(false, Ordering::Relaxed);
                        continue;
                    }
                };

            let response = trim_voice_response(&response);
            if response.is_empty() {
                signals.response_active.store(false, Ordering::Relaxed);
                continue;
            }
            let response = HumanizedSpeech::from_response(&response, state.humanization.tone).text;
            if let Some(tracker) = call_tracker.as_ref() {
                tracker.record_assistant_text(response.clone());
            }

            {
                let mut history = history.lock().await;
                history.push(("assistant".to_string(), response.clone()));
                keep_recent_history(&mut history, 10);
            }

            if generation != signals.response_generation.load(Ordering::Relaxed) {
                continue;
            }

            *signals.response_requested_at.lock().await = Some((generation, Instant::now()));

            if let Err(e) = cartesia_tx.send(SplitControl::Speak {
                generation,
                text: response,
            }) {
                warn!("failed sending text to Cartesia worker: {}", e);
                signals.response_active.store(false, Ordering::Relaxed);
            }
        }
    });
}

fn spawn_cartesia_tts(
    state: AppState,
    api_key: String,
    stream_sid: String,
    twilio_out_tx: mpsc::Sender<Value>,
    signals: RealtimeSignals,
    mut control_rx: mpsc::UnboundedReceiver<SplitControl>,
) {
    tokio::spawn(async move {
        let url = format!(
            "wss://api.cartesia.ai/tts/websocket?cartesia_version={}",
            CARTESIA_VERSION
        );
        let mut request = match url.into_client_request() {
            Ok(request) => request,
            Err(e) => {
                error!("failed building Cartesia request: {}", e);
                return;
            }
        };
        if let Ok(header_value) = HeaderValue::from_str(&api_key) {
            request.headers_mut().insert("X-API-Key", header_value);
        }
        request.headers_mut().insert(
            "Cartesia-Version",
            HeaderValue::from_static(CARTESIA_VERSION),
        );

        info!("connecting to Cartesia streaming TTS");
        let (ws, _) = match connect_async(request).await {
            Ok(connected) => connected,
            Err(e) => {
                error!("failed connecting to Cartesia: {}", e);
                return;
            }
        };
        info!("connected to Cartesia streaming TTS");

        let (mut cartesia_sender, mut cartesia_receiver) = ws.split();
        let active_context = Arc::new(Mutex::new(None::<String>));
        let active_humanizer = Arc::new(Mutex::new(None::<AudioHumanizer>));
        let active_context_for_reader = active_context.clone();
        let active_humanizer_for_reader = active_humanizer.clone();
        let signals_for_reader = signals.clone();
        let twilio_out_tx_for_reader = twilio_out_tx.clone();
        let stream_sid_for_reader = stream_sid.clone();

        tokio::spawn(async move {
            let mut ulaw_buffer = Vec::<u8>::new();

            while let Some(msg) = cartesia_receiver.next().await {
                let msg = match msg {
                    Ok(msg) => msg,
                    Err(e) => {
                        warn!("Cartesia receive error: {}", e);
                        break;
                    }
                };

                let text = match msg {
                    TungsteniteMessage::Text(text) => text,
                    TungsteniteMessage::Close(_) => {
                        info!("Cartesia websocket closed");
                        break;
                    }
                    _ => continue,
                };

                let value: Value = match serde_json::from_str(&text) {
                    Ok(value) => value,
                    Err(e) => {
                        warn!("bad Cartesia JSON: {}", e);
                        continue;
                    }
                };

                match value
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                {
                    "chunk" => {
                        let Some(context_id) = value.get("context_id").and_then(|v| v.as_str())
                        else {
                            continue;
                        };
                        let active_context = active_context_for_reader.lock().await;
                        if active_context.as_deref() != Some(context_id) {
                            continue;
                        }
                        drop(active_context);

                        let Some(data) = value.get("data").and_then(|v| v.as_str()) else {
                            continue;
                        };
                        let pcm = match general_purpose::STANDARD.decode(data) {
                            Ok(pcm) => pcm,
                            Err(e) => {
                                warn!("bad Cartesia audio chunk: {}", e);
                                continue;
                            }
                        };
                        let mut voice_samples = pcm_s16le_to_i16(&pcm);
                        if let Some(humanizer) = active_humanizer_for_reader.lock().await.as_mut() {
                            voice_samples = humanizer.process_chunk(
                                &voice_samples,
                                signals_for_reader.interrupted.load(Ordering::Relaxed),
                            );
                        }
                        ulaw_buffer.extend(pcm_i16_to_ulaw(&voice_samples));
                        let generation =
                            split_generation_from_context(context_id).unwrap_or_else(|| {
                                signals_for_reader
                                    .response_generation
                                    .load(Ordering::Relaxed)
                            });

                        if generation
                            != signals_for_reader
                                .response_generation
                                .load(Ordering::Relaxed)
                        {
                            continue;
                        }

                        while ulaw_buffer.len() >= TWILIO_ULAW_FRAME_BYTES {
                            let frame = ulaw_buffer
                                .drain(..TWILIO_ULAW_FRAME_BYTES)
                                .collect::<Vec<_>>();
                            if queue_twilio_ulaw_frame(
                                &twilio_out_tx_for_reader,
                                &stream_sid_for_reader,
                                &signals_for_reader,
                                generation,
                                frame,
                                state.soft_audio_limit_ms,
                            )
                            .await
                            .is_err()
                            {
                                return;
                            }
                        }
                    }
                    "done" => {
                        let context_id = value.get("context_id").and_then(|v| v.as_str());
                        let generation = match context_id {
                            Some(context_id) => {
                                let mut active_context = active_context_for_reader.lock().await;
                                if active_context.as_deref() != Some(context_id) {
                                    if active_context.is_none() {
                                        ulaw_buffer.clear();
                                        *active_humanizer_for_reader.lock().await = None;
                                    }
                                    continue;
                                }
                                *active_context = None;
                                split_generation_from_context(context_id).unwrap_or_else(|| {
                                    signals_for_reader
                                        .response_generation
                                        .load(Ordering::Relaxed)
                                })
                            }
                            None => {
                                let mut active_context = active_context_for_reader.lock().await;
                                let Some(active_context_id) = active_context.take() else {
                                    ulaw_buffer.clear();
                                    *active_humanizer_for_reader.lock().await = None;
                                    continue;
                                };
                                split_generation_from_context(&active_context_id).unwrap_or_else(
                                    || {
                                        signals_for_reader
                                            .response_generation
                                            .load(Ordering::Relaxed)
                                    },
                                )
                            }
                        };

                        if generation
                            != signals_for_reader
                                .response_generation
                                .load(Ordering::Relaxed)
                        {
                            ulaw_buffer.clear();
                            *active_humanizer_for_reader.lock().await = None;
                            continue;
                        }

                        if let Some(humanizer) = active_humanizer_for_reader.lock().await.as_mut() {
                            ulaw_buffer.extend(pcm_i16_to_ulaw(&humanizer.finish()));
                        }
                        *active_humanizer_for_reader.lock().await = None;

                        if !ulaw_buffer.is_empty() {
                            let frame = std::mem::take(&mut ulaw_buffer);
                            if queue_twilio_ulaw_frame(
                                &twilio_out_tx_for_reader,
                                &stream_sid_for_reader,
                                &signals_for_reader,
                                generation,
                                frame,
                                state.soft_audio_limit_ms,
                            )
                            .await
                            .is_err()
                            {
                                return;
                            }
                        }

                        signals_for_reader
                            .response_active
                            .store(false, Ordering::Relaxed);
                        info!(
                            "playback_metrics split_response_done generated_audio={}ms played_audio={}ms queued_audio={}ms pending_marks={}",
                            signals_for_reader.generated_audio_ms.load(Ordering::Relaxed),
                            signals_for_reader.played_audio_ms.load(Ordering::Relaxed),
                            signals_for_reader.queued_audio_ms.load(Ordering::Relaxed),
                            signals_for_reader.pending_playback_marks.load(Ordering::Relaxed)
                        );
                    }
                    "error" => {
                        let context_id = value.get("context_id").and_then(|v| v.as_str());
                        let active_context = active_context_for_reader.lock().await;
                        let stale_context =
                            context_id.is_some() && active_context.as_deref() != context_id;
                        drop(active_context);

                        if stale_context {
                            debug!("stale Cartesia context error ignored: {}", value);
                        } else {
                            error!("Cartesia error: {}", value);
                            signals_for_reader
                                .response_active
                                .store(false, Ordering::Relaxed);
                        }
                    }
                    _ => {}
                }
            }
        });

        while let Some(command) = control_rx.recv().await {
            match command {
                SplitControl::Speak { generation, text } => {
                    let Some(voice_id) = state.cartesia_voice_id.as_ref() else {
                        error!("CARTESIA_VOICE_ID missing");
                        continue;
                    };
                    let context_id = format!(
                        "twilio-{}-gen-{}",
                        sanitize_context_id(&stream_sid),
                        generation
                    );
                    *active_context.lock().await = Some(context_id.clone());
                    let speech = HumanizedSpeech::from_spoken_text(&text, state.humanization.tone);
                    *active_humanizer.lock().await =
                        Some(AudioHumanizer::new(state.humanization.clone(), &speech));

                    let request = json!({
                        "model_id": state.cartesia_model,
                        "transcript": speech.text,
                        "voice": {
                            "mode": "id",
                            "id": voice_id
                        },
                        "language": state.cartesia_language,
                        "context_id": context_id,
                        "output_format": {
                            "container": "raw",
                            "encoding": "pcm_s16le",
                            "sample_rate": 8000
                        },
                        "add_timestamps": false,
                        "continue": false
                    });

                    if let Err(e) = cartesia_sender
                        .send(TungsteniteMessage::Text(request.to_string()))
                        .await
                    {
                        warn!("failed sending text to Cartesia: {}", e);
                        break;
                    }
                }
                SplitControl::Cancel { context_id } => {
                    let context_id = match context_id {
                        Some(context_id) => context_id,
                        None => active_context.lock().await.clone().unwrap_or_default(),
                    };
                    if context_id.is_empty() {
                        continue;
                    }
                    let request = json!({
                        "context_id": context_id,
                        "cancel": true
                    });
                    *active_context.lock().await = None;
                    if let Some(humanizer) = active_humanizer.lock().await.as_mut() {
                        humanizer.clear();
                    }
                    *active_humanizer.lock().await = None;
                    if let Err(e) = cartesia_sender
                        .send(TungsteniteMessage::Text(request.to_string()))
                        .await
                    {
                        warn!("failed sending Cartesia cancel: {}", e);
                        break;
                    }
                }
            }
        }
    });
}

async fn queue_twilio_ulaw_frame(
    twilio_out_tx: &mpsc::Sender<Value>,
    stream_sid: &str,
    signals: &RealtimeSignals,
    generation: u64,
    frame: Vec<u8>,
    soft_audio_limit_ms: u64,
) -> Result<()> {
    let audio_ms = (frame.len() as u64 / ULAW_BYTES_PER_MS as u64).max(1);
    let payload = general_purpose::STANDARD.encode(frame);
    let generated_ms = signals
        .generated_audio_ms
        .fetch_add(audio_ms, Ordering::Relaxed)
        + audio_ms;

    let mut requested_at = signals.response_requested_at.lock().await;
    if let Some((requested_generation, started_at)) = *requested_at {
        if requested_generation == generation {
            info!(
                "latency split_tts_to_first_audio={}ms generation={}",
                started_at.elapsed().as_millis(),
                generation
            );
            *requested_at = None;
        }
    }
    drop(requested_at);

    if generated_ms > soft_audio_limit_ms {
        debug!(
            "split soft response audio limit exceeded generated={}ms limit={}ms",
            generated_ms, soft_audio_limit_ms
        );
    }

    let twilio_media = json!({
        "event": "media",
        "streamSid": stream_sid,
        "generation": generation,
        "audio_ms": audio_ms,
        "media": {
            "payload": payload
        }
    });

    twilio_out_tx
        .send(twilio_media)
        .await
        .context("failed queueing Cartesia audio to Twilio")?;
    signals.assistant_speaking.store(true, Ordering::Relaxed);

    tokio::time::sleep(Duration::from_millis(audio_ms)).await;

    Ok(())
}

async fn stream_openai_text_response(
    client: &reqwest::Client,
    state: &AppState,
    history: Arc<Mutex<Vec<(String, String)>>>,
    generation: u64,
) -> Result<String> {
    let backlog_ms = state.backlog_high_water_ms.min(state.soft_audio_limit_ms);
    let budget = response_budget(
        state.max_response_tokens,
        backlog_ms,
        state.backlog_low_water_ms,
        state.backlog_high_water_ms,
    );
    let mut instructions =
        response_instructions_for_budget(&state.response_instructions, budget, 0);
    instructions.push('\n');
    instructions.push_str(&state.persona_instructions);

    let input = {
        let history = history.lock().await;
        history
            .iter()
            .map(|(role, content)| {
                json!({
                    "role": role,
                    "content": content
                })
            })
            .collect::<Vec<_>>()
    };

    let request_started_at = Instant::now();
    let response = client
        .post("https://api.openai.com/v1/responses")
        .bearer_auth(&state.openai_api_key)
        .json(&json!({
            "model": state.openai_text_model,
            "instructions": instructions,
            "input": input,
            "max_output_tokens": budget.max_tokens,
            "stream": true
        }))
        .send()
        .await
        .context("failed sending OpenAI text request")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("OpenAI text HTTP {}: {}", status, body));
    }

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut output = String::new();
    let mut saw_first_delta = false;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("failed reading OpenAI text stream")?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(line_end) = buffer.find('\n') {
            let line = buffer[..line_end].trim().to_string();
            buffer.drain(..=line_end);

            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }

            let value: Value = match serde_json::from_str(data) {
                Ok(value) => value,
                Err(_) => continue,
            };
            match value
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
            {
                "response.output_text.delta" => {
                    if !saw_first_delta {
                        info!(
                            "latency split_llm_request_to_first_text={}ms generation={}",
                            request_started_at.elapsed().as_millis(),
                            generation
                        );
                        saw_first_delta = true;
                    }
                    if let Some(delta) = value.get("delta").and_then(|v| v.as_str()) {
                        output.push_str(delta);
                        if completed_short_voice_turn(&output) {
                            return Ok(output);
                        }
                    }
                }
                "response.completed" => return Ok(output),
                "error" | "response.failed" => {
                    return Err(anyhow!("OpenAI text stream error: {}", value));
                }
                _ => {}
            }
        }
    }

    Ok(output)
}

fn completed_short_voice_turn(text: &str) -> bool {
    let trimmed = text.trim();
    let word_count = trimmed.split_whitespace().count();

    word_count >= 6
        && (trimmed.ends_with('?')
            || trimmed.ends_with('.')
            || trimmed.ends_with('!')
            || word_count >= 14)
}

fn trim_voice_response(text: &str) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut words = normalized.split_whitespace().take(10).collect::<Vec<_>>();
    let mut response = words.join(" ");

    if response.len() < normalized.len() {
        response = response
            .trim_end_matches(|c: char| c == ',' || c == ';' || c == ':')
            .to_string();
        if !response.ends_with('?') && !response.ends_with('.') && !response.ends_with('!') {
            response.push('.');
        }
    }

    words.clear();
    response
}

fn keep_recent_history(history: &mut Vec<(String, String)>, max_messages: usize) {
    if history.len() > max_messages {
        let remove_count = history.len() - max_messages;
        history.drain(..remove_count);
    }
}

fn deepgram_transcript(value: &Value) -> &str {
    value
        .get("channel")
        .and_then(|channel| channel.get("alternatives"))
        .and_then(|alternatives| alternatives.as_array())
        .and_then(|alternatives| alternatives.first())
        .and_then(|alternative| alternative.get("transcript"))
        .and_then(|transcript| transcript.as_str())
        .unwrap_or_default()
}

fn predictive_candidate(text: &str) -> bool {
    let trimmed = text.trim();
    let word_count = trimmed.split_whitespace().count();

    if word_count < 4 {
        return false;
    }

    word_count >= 4
        || trimmed.ends_with('?')
        || trimmed.ends_with('.')
        || trimmed.ends_with('!')
        || contains_predictive_intent(trimmed)
}

fn contains_predictive_intent(text: &str) -> bool {
    let lower = text.to_lowercase();
    [
        "calling from",
        "your account",
        "your bank",
        "transaction",
        "blocked",
        "urgent",
        "fraud",
        "employee number",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn sanitize_context_id(value: &str) -> String {
    value
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect()
}

fn split_generation_from_context(context_id: &str) -> Option<u64> {
    context_id.rsplit_once("-gen-")?.1.parse().ok()
}
