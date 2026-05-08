use crate::audio::{
    pcm_i16_to_ulaw, ulaw_base64_duration_ms, ulaw_bytes_to_pcm_i16, MEDIA_FRAME_MS,
};
use crate::config::{AppState, TurnTakingConfig};
use crate::humanize::{AudioHumanizer, HumanizedSpeech};
use crate::response_policy::{response_budget, response_instructions_for_budget};
use crate::signals::RealtimeSignals;
use crate::tracking::CallTracker;
use anyhow::{Context, Result};
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

const OPENAI_TURN_DETECTION: &str = "server_vad";

#[derive(Clone)]
pub(crate) struct OpenAiHandle {
    tx: mpsc::Sender<Value>,
    response_instructions: Arc<String>,
    max_response_tokens: u32,
    soft_audio_limit_ms: u64,
    hard_audio_limit_ms: u64,
    backlog_low_water_ms: u64,
    backlog_high_water_ms: u64,
    queued_audio_ms: Arc<AtomicU64>,
}

impl OpenAiHandle {
    async fn send(&self, value: Value) {
        if let Err(e) = self.tx.send(value).await {
            warn!("failed to send to OpenAI queue: {}", e);
        }
    }

    pub(crate) async fn append_audio_ulaw(&self, b64_payload: &str) {
        self.send(json!({
            "type": "input_audio_buffer.append",
            "audio": b64_payload
        }))
        .await;
    }

    async fn create_response(&self) {
        let backlog_ms = self.queued_audio_ms.load(Ordering::Relaxed);
        let budget = response_budget(
            self.max_response_tokens,
            backlog_ms,
            self.backlog_low_water_ms,
            self.backlog_high_water_ms,
        );
        let instructions = response_instructions_for_budget(
            self.response_instructions.as_str(),
            budget,
            backlog_ms,
        );

        self.send(json!({
            "type": "response.create",
            "response": {
                "modalities": ["audio", "text"],
                "instructions": instructions,
                "max_output_tokens": budget.max_tokens
            }
        }))
        .await;
    }

    pub(crate) async fn cancel_response(&self) {
        self.send(json!({
            "type": "response.cancel"
        }))
        .await;
    }
}

pub(crate) async fn connect_openai_realtime(
    state: AppState,
    stream_sid: String,
    twilio_out_tx: mpsc::Sender<Value>,
    signals: RealtimeSignals,
    call_tracker: Option<CallTracker>,
) -> Result<OpenAiHandle> {
    let url = format!(
        "wss://api.openai.com/v1/realtime?model={}",
        state.openai_model
    );

    let mut request = url.into_client_request()?;

    let auth_value = format!("Bearer {}", state.openai_api_key);

    request
        .headers_mut()
        .insert("Authorization", HeaderValue::from_str(&auth_value)?);

    request
        .headers_mut()
        .insert("OpenAI-Beta", HeaderValue::from_static("realtime=v1"));

    info!("connecting to OpenAI Realtime");

    let (ws, _) = connect_async(request)
        .await
        .context("failed connecting to OpenAI realtime")?;

    info!("connected to OpenAI Realtime");

    let (mut oai_sender, mut oai_receiver) = ws.split();

    let (oai_tx, mut oai_rx) = mpsc::channel::<Value>(128);
    let turn_config = state.turn.clone();

    tokio::spawn(async move {
        while let Some(value) = oai_rx.recv().await {
            let text = value.to_string();

            if let Err(e) = oai_sender.send(TungsteniteMessage::Text(text)).await {
                warn!("failed sending to OpenAI: {}", e);
                break;
            }
        }
    });

    let handle = OpenAiHandle {
        tx: oai_tx.clone(),
        response_instructions: Arc::new(state.response_instructions.clone()),
        max_response_tokens: state.max_response_tokens,
        soft_audio_limit_ms: state.soft_audio_limit_ms,
        hard_audio_limit_ms: state.hard_audio_limit_ms,
        backlog_low_water_ms: state.backlog_low_water_ms,
        backlog_high_water_ms: state.backlog_high_water_ms,
        queued_audio_ms: signals.queued_audio_ms.clone(),
    };

    handle
        .send(json!({
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "voice": state.openai_voice,
                "instructions": state.persona_instructions,
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "turn_detection": {
                    "type": OPENAI_TURN_DETECTION,
                    "threshold": turn_config.vad_threshold,
                    "prefix_padding_ms": turn_config.vad_prefix_padding_ms,
                    "silence_duration_ms": turn_config.vad_silence_duration_ms,
                    "create_response": false,
                    "interrupt_response": false
                },
                "temperature": 0.7
            }
        }))
        .await;

    let twilio_out_tx_for_reader = twilio_out_tx.clone();
    let stream_sid_for_reader = stream_sid.clone();
    let assistant_speaking_for_reader = signals.assistant_speaking.clone();
    let response_active_for_reader = signals.response_active.clone();
    let pending_playback_marks_for_reader = signals.pending_playback_marks.clone();
    let generated_audio_ms_for_reader = signals.generated_audio_ms.clone();
    let user_speech_generation_for_reader = signals.user_speech_generation.clone();
    let current_turn_speech_ms_for_reader = signals.current_turn_speech_ms.clone();
    let openai_speech_started_ms_for_reader = signals.openai_speech_started_ms.clone();
    let handle_for_reader = handle.clone();
    let turn_config_for_reader = turn_config.clone();
    let humanization_config_for_reader = state.humanization.clone();
    let active_humanizer = Arc::new(Mutex::new(None::<AudioHumanizer>));
    let active_humanizer_for_reader = active_humanizer.clone();

    tokio::spawn(async move {
        let mut last_audio_at = Instant::now();

        while let Some(msg) = oai_receiver.next().await {
            let msg = match msg {
                Ok(m) => m,
                Err(e) => {
                    warn!("OpenAI receive error: {}", e);
                    break;
                }
            };

            let text = match msg {
                TungsteniteMessage::Text(t) => t,
                TungsteniteMessage::Close(_) => {
                    info!("OpenAI websocket closed");
                    break;
                }
                _ => continue,
            };

            let value: Value = match serde_json::from_str(&text) {
                Ok(v) => v,
                Err(e) => {
                    warn!("bad OpenAI JSON: {}", e);
                    continue;
                }
            };

            let event_type = value
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or_default();

            match event_type {
                "response.created" => {
                    response_active_for_reader.store(true, Ordering::Relaxed);
                    *active_humanizer_for_reader.lock().await = Some(AudioHumanizer::new(
                        humanization_config_for_reader.clone(),
                        &HumanizedSpeech::from_spoken_text("", humanization_config_for_reader.tone),
                    ));
                    if let Some(response_id) = response_id(&value) {
                        *signals.active_response_id.lock().await = Some(response_id.to_string());
                    }
                }

                "input_audio_buffer.speech_started" => {
                    user_speech_generation_for_reader.fetch_add(1, Ordering::Relaxed);
                    current_turn_speech_ms_for_reader.store(0, Ordering::Relaxed);
                    let openai_start_ms = value
                        .get("audio_start_ms")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    openai_speech_started_ms_for_reader.store(openai_start_ms, Ordering::Relaxed);

                    if response_active_for_reader.load(Ordering::Relaxed)
                        && !assistant_speaking_for_reader.load(Ordering::Relaxed)
                    {
                        response_active_for_reader.store(false, Ordering::Relaxed);
                        *signals.response_requested_at.lock().await = None;
                        *signals.active_response_id.lock().await = None;
                        signals.response_generation.fetch_add(1, Ordering::Relaxed);
                        handle_for_reader.cancel_response().await;
                    }

                    debug!("OpenAI VAD speech started: {}", value);
                }

                "input_audio_buffer.speech_stopped" => {
                    let stopped_generation =
                        user_speech_generation_for_reader.load(Ordering::Relaxed);
                    let local_turn_speech_ms =
                        current_turn_speech_ms_for_reader.load(Ordering::Relaxed);
                    let openai_turn_speech_ms = openai_turn_speech_ms(
                        &value,
                        openai_speech_started_ms_for_reader.load(Ordering::Relaxed),
                    );
                    let turn_speech_ms = openai_turn_speech_ms;

                    if turn_speech_ms < turn_config_for_reader.min_response_speech_ms {
                        info!(
                            "ignoring short/noisy VAD stop turn_speech={}ms local={}ms openai={}ms minimum={}ms",
                            turn_speech_ms,
                            local_turn_speech_ms,
                            openai_turn_speech_ms,
                            turn_config_for_reader.min_response_speech_ms
                        );
                        continue;
                    }

                    let grace_ms = response_grace_ms(turn_speech_ms, &turn_config_for_reader);
                    let speech_stopped_at = Instant::now();
                    let generation_for_task = user_speech_generation_for_reader.clone();
                    let response_active_for_task = response_active_for_reader.clone();
                    let handle_for_task = handle_for_reader.clone();
                    let signals_for_task = signals.clone();

                    tokio::spawn(async move {
                        tokio::time::sleep(Duration::from_millis(grace_ms)).await;

                        let speech_resumed =
                            generation_for_task.load(Ordering::Relaxed) != stopped_generation;

                        if speech_resumed || response_active_for_task.load(Ordering::Relaxed) {
                            return;
                        }

                        let generation = signals_for_task
                            .response_generation
                            .fetch_add(1, Ordering::Relaxed)
                            + 1;
                        signals_for_task
                            .generated_audio_ms
                            .store(0, Ordering::Relaxed);
                        signals_for_task.played_audio_ms.store(0, Ordering::Relaxed);
                        signals_for_task.queued_audio_ms.store(0, Ordering::Relaxed);
                        signals_for_task.interrupted.store(false, Ordering::Relaxed);
                        *signals_for_task.response_requested_at.lock().await =
                            Some((generation, Instant::now()));
                        *signals_for_task.active_response_id.lock().await = None;

                        info!(
                            "latency vad_stop_to_response_create={}ms turn_speech={}ms grace={}ms generation={}",
                            speech_stopped_at.elapsed().as_millis(),
                            turn_speech_ms,
                            grace_ms,
                            generation
                        );

                        response_active_for_task.store(true, Ordering::Relaxed);
                        handle_for_task.create_response().await;
                    });

                    debug!(
                        "OpenAI VAD speech stopped after {}ms local speech; response grace {}ms: {}",
                        turn_speech_ms, grace_ms, value
                    );
                }

                "response.audio.delta" => {
                    if let Some(delta) = value.get("delta").and_then(|v| v.as_str()) {
                        if let Some(delta_response_id) = response_id(&value) {
                            let mut active_response_id = signals.active_response_id.lock().await;

                            match active_response_id.as_deref() {
                                Some(active_id) if active_id != delta_response_id => {
                                    debug!(
                                        "dropping stale audio delta for response {}",
                                        delta_response_id
                                    );
                                    continue;
                                }
                                None => {
                                    *active_response_id = Some(delta_response_id.to_string());
                                }
                                _ => {}
                            }
                        }

                        let generation = signals.response_generation.load(Ordering::Relaxed);
                        let transformed_delta =
                            transform_realtime_delta(delta, &active_humanizer_for_reader, &signals)
                                .await
                                .unwrap_or_else(|| delta.to_string());
                        let audio_ms = ulaw_base64_duration_ms(&transformed_delta)
                            .unwrap_or(MEDIA_FRAME_MS as u64);
                        let generated_ms = generated_audio_ms_for_reader
                            .fetch_add(audio_ms, Ordering::Relaxed)
                            + audio_ms;

                        if handle_for_reader.hard_audio_limit_ms > 0
                            && generated_ms >= handle_for_reader.hard_audio_limit_ms
                        {
                            info!(
                                "hard response audio limit reached generated={}ms limit={}ms; cancelling",
                                generated_ms, handle_for_reader.hard_audio_limit_ms
                            );
                            handle_for_reader.cancel_response().await;
                            response_active_for_reader.store(false, Ordering::Relaxed);
                            *signals.active_response_id.lock().await = None;
                            continue;
                        } else if generated_ms >= handle_for_reader.soft_audio_limit_ms {
                            debug!(
                                "soft response audio limit exceeded generated={}ms limit={}ms",
                                generated_ms, handle_for_reader.soft_audio_limit_ms
                            );
                        }

                        let mut requested_at = signals.response_requested_at.lock().await;
                        if let Some((requested_generation, started_at)) = *requested_at {
                            if requested_generation == generation {
                                info!(
                                    "latency response_create_to_first_audio={}ms generation={}",
                                    started_at.elapsed().as_millis(),
                                    generation
                                );
                                *requested_at = None;
                            }
                        }

                        let twilio_media = json!({
                            "event": "media",
                            "streamSid": stream_sid_for_reader,
                            "generation": generation,
                            "audio_ms": audio_ms,
                            "media": {
                                "payload": transformed_delta
                            }
                        });

                        if let Err(e) = twilio_out_tx_for_reader.send(twilio_media).await {
                            warn!("failed queueing Twilio audio: {}", e);
                            break;
                        }

                        assistant_speaking_for_reader.store(true, Ordering::Relaxed);
                        last_audio_at = Instant::now();
                    }
                }

                "response.done" => {
                    info!(
                        "playback_metrics response_done generated_audio={}ms played_audio={}ms queued_audio={}ms pending_marks={}",
                        signals.generated_audio_ms.load(Ordering::Relaxed),
                        signals.played_audio_ms.load(Ordering::Relaxed),
                        signals.queued_audio_ms.load(Ordering::Relaxed),
                        pending_playback_marks_for_reader.load(Ordering::Relaxed)
                    );
                    response_active_for_reader.store(false, Ordering::Relaxed);
                    *active_humanizer_for_reader.lock().await = None;
                    *signals.response_requested_at.lock().await = None;
                    *signals.active_response_id.lock().await = None;
                    if pending_playback_marks_for_reader.load(Ordering::Relaxed) == 0 {
                        assistant_speaking_for_reader.store(false, Ordering::Relaxed);
                    }
                }

                "conversation.item.input_audio_transcription.completed" => {
                    if let Some(transcript) = value.get("transcript").and_then(|v| v.as_str()) {
                        if let Some(tracker) = call_tracker.as_ref() {
                            tracker.record_user_text(transcript.to_string());
                        }
                    }
                }

                "response.audio_transcript.done" | "response.output_text.done" => {
                    if let Some(transcript) = value
                        .get("transcript")
                        .or_else(|| value.get("text"))
                        .and_then(|v| v.as_str())
                    {
                        if let Some(tracker) = call_tracker.as_ref() {
                            tracker.record_assistant_text(transcript.to_string());
                        }
                    }
                }

                "response.cancelled" => {
                    debug!("OpenAI response cancelled");
                    response_active_for_reader.store(false, Ordering::Relaxed);
                    *signals.response_requested_at.lock().await = None;
                    *signals.active_response_id.lock().await = None;
                    pending_playback_marks_for_reader.store(0, Ordering::Relaxed);
                    assistant_speaking_for_reader.store(false, Ordering::Relaxed);
                    if let Some(humanizer) = active_humanizer_for_reader.lock().await.as_mut() {
                        humanizer.clear();
                    }
                    *active_humanizer_for_reader.lock().await = None;
                }

                "error" => {
                    error!("OpenAI error: {}", value);
                    *signals.response_requested_at.lock().await = None;
                    *signals.active_response_id.lock().await = None;
                    response_active_for_reader.store(false, Ordering::Relaxed);
                    *active_humanizer_for_reader.lock().await = None;
                }

                other => {
                    debug!("OpenAI event {}: {}", other, value);
                }
            }

            if last_audio_at.elapsed() > Duration::from_millis(700)
                && pending_playback_marks_for_reader.load(Ordering::Relaxed) == 0
            {
                assistant_speaking_for_reader.store(false, Ordering::Relaxed);
            }
        }
    });

    Ok(handle)
}

fn response_grace_ms(turn_speech_ms: u64, config: &TurnTakingConfig) -> u64 {
    if turn_speech_ms >= 1_500 {
        config.long_turn_grace_ms
    } else if turn_speech_ms >= 600 {
        config.normal_turn_grace_ms
    } else {
        config.short_turn_grace_ms
    }
}

fn openai_turn_speech_ms(value: &Value, started_ms: u64) -> u64 {
    let Some(stopped_ms) = value.get("audio_end_ms").and_then(|v| v.as_u64()) else {
        return 0;
    };

    stopped_ms.saturating_sub(started_ms)
}

async fn transform_realtime_delta(
    delta: &str,
    active_humanizer: &Arc<Mutex<Option<AudioHumanizer>>>,
    signals: &RealtimeSignals,
) -> Option<String> {
    let ulaw = general_purpose::STANDARD.decode(delta).ok()?;
    let pcm = ulaw_bytes_to_pcm_i16(&ulaw);
    let mut humanizer = active_humanizer.lock().await;
    let humanizer = humanizer.as_mut()?;
    let transformed = humanizer.process_chunk(&pcm, signals.interrupted.load(Ordering::Relaxed));
    if transformed.is_empty() {
        return None;
    }
    Some(general_purpose::STANDARD.encode(pcm_i16_to_ulaw(&transformed)))
}

fn response_id(value: &Value) -> Option<&str> {
    value
        .get("response_id")
        .and_then(|v| v.as_str())
        .or_else(|| {
            value
                .get("response")
                .and_then(|r| r.get("id"))
                .and_then(|v| v.as_str())
        })
}
