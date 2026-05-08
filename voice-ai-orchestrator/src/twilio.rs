use crate::audio::{parse_playback_mark, WebRtcBargeInVad, MEDIA_FRAME_MS};
use crate::config::{AppState, OrchestratorStack, TurnTakingConfig};
use crate::humanize::humanization_metrics;
use crate::openai_realtime::{connect_openai_realtime, OpenAiHandle};
use crate::signals::RealtimeSignals;
use crate::split_providers::{connect_split_providers, SplitControl};
use crate::tracking::{CallStart, CallTracker, FinalCallMetrics};
use anyhow::{anyhow, Result};
use axum::extract::ws::{Message as AxumMessage, WebSocket};
use base64::{engine::general_purpose, Engine as _};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc,
};
use std::time::Instant;
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, info, warn};

struct AudioFrame {
    b64_payload: String,
}

enum CallPipeline {
    OpenAiRealtime {
        audio_tx: mpsc::UnboundedSender<AudioFrame>,
        interrupt_tx: mpsc::UnboundedSender<Vec<u8>>,
    },
    SplitProviders {
        audio_tx: mpsc::UnboundedSender<Vec<u8>>,
        interrupt_tx: mpsc::UnboundedSender<Vec<u8>>,
    },
}

pub(crate) async fn handle_twilio_socket(socket: WebSocket, state: AppState) -> Result<()> {
    let call_started_at = Instant::now();

    let assistant_speaking = Arc::new(AtomicBool::new(false));
    let response_active = Arc::new(AtomicBool::new(false));
    let interrupted = Arc::new(AtomicBool::new(false));
    let pending_playback_marks = Arc::new(AtomicUsize::new(0));
    let response_generation = Arc::new(AtomicU64::new(1));
    let generated_audio_ms = Arc::new(AtomicU64::new(0));
    let played_audio_ms = Arc::new(AtomicU64::new(0));
    let queued_audio_ms = Arc::new(AtomicU64::new(0));
    let active_response_id = Arc::new(Mutex::new(None));
    let response_requested_at = Arc::new(Mutex::new(None));
    let user_speech_generation = Arc::new(AtomicU64::new(0));
    let current_turn_speech_ms = Arc::new(AtomicU64::new(0));
    let openai_speech_started_ms = Arc::new(AtomicU64::new(0));
    let realtime_signals = RealtimeSignals {
        assistant_speaking: assistant_speaking.clone(),
        response_active: response_active.clone(),
        interrupted: interrupted.clone(),
        pending_playback_marks: pending_playback_marks.clone(),
        response_generation: response_generation.clone(),
        generated_audio_ms: generated_audio_ms.clone(),
        played_audio_ms: played_audio_ms.clone(),
        queued_audio_ms: queued_audio_ms.clone(),
        active_response_id: active_response_id.clone(),
        response_requested_at: response_requested_at.clone(),
        user_speech_generation: user_speech_generation.clone(),
        current_turn_speech_ms: current_turn_speech_ms.clone(),
        openai_speech_started_ms: openai_speech_started_ms.clone(),
    };

    let (mut twilio_sender, mut twilio_receiver) = socket.split();

    let (twilio_out_tx, mut twilio_out_rx) = mpsc::channel::<Value>(128);

    let assistant_speaking_for_sender = assistant_speaking.clone();
    let pending_playback_marks_for_sender = pending_playback_marks.clone();
    let response_generation_for_sender = response_generation.clone();
    let played_audio_ms_for_sender = played_audio_ms.clone();
    let queued_audio_ms_for_sender = queued_audio_ms.clone();
    let mark_seq = Arc::new(AtomicU64::new(1));
    let mark_seq_for_sender = mark_seq.clone();

    tokio::spawn(async move {
        while let Some(msg) = twilio_out_rx.recv().await {
            let is_media = msg.get("event").and_then(|v| v.as_str()) == Some("media");
            let is_clear = msg.get("event").and_then(|v| v.as_str()) == Some("clear");
            let generation = msg.get("generation").and_then(|v| v.as_u64()).unwrap_or(0);
            let stream_sid = msg
                .get("streamSid")
                .and_then(|v| v.as_str())
                .map(str::to_owned);

            if is_media && generation != response_generation_for_sender.load(Ordering::Relaxed) {
                continue;
            }

            let audio_ms = msg
                .get("audio_ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(MEDIA_FRAME_MS as u64);
            let mut outbound_msg = msg;
            if let Some(obj) = outbound_msg.as_object_mut() {
                obj.remove("generation");
                obj.remove("audio_ms");
            }
            let text = outbound_msg.to_string();

            if let Err(e) = twilio_sender.send(AxumMessage::Text(text)).await {
                warn!("failed sending to Twilio: {}", e);
                break;
            }

            if is_media {
                assistant_speaking_for_sender.store(true, Ordering::Relaxed);
                queued_audio_ms_for_sender.fetch_add(audio_ms, Ordering::Relaxed);

                if let Some(stream_sid) = stream_sid {
                    let mark_name = format!(
                        "assistant-{}-{}-{}",
                        generation,
                        audio_ms,
                        mark_seq_for_sender.fetch_add(1, Ordering::Relaxed)
                    );
                    let mark = json!({
                        "event": "mark",
                        "streamSid": stream_sid,
                        "mark": {
                            "name": mark_name
                        }
                    });

                    pending_playback_marks_for_sender.fetch_add(1, Ordering::Relaxed);

                    if let Err(e) = twilio_sender
                        .send(AxumMessage::Text(mark.to_string()))
                        .await
                    {
                        warn!("failed sending Twilio playback mark: {}", e);
                        break;
                    }
                };
            } else if is_clear {
                pending_playback_marks_for_sender.store(0, Ordering::Relaxed);
                queued_audio_ms_for_sender.store(0, Ordering::Relaxed);
                played_audio_ms_for_sender.store(0, Ordering::Relaxed);
            }
        }
    });

    let mut pipeline: Option<CallPipeline> = None;
    let mut call_tracker: Option<CallTracker> = None;

    info!("twilio websocket connected");

    while let Some(msg) = twilio_receiver.next().await {
        let msg = match msg {
            Ok(m) => m,
            Err(e) => {
                warn!("twilio ws receive error: {}", e);
                break;
            }
        };

        let text = match msg {
            AxumMessage::Text(t) => t,
            AxumMessage::Close(_) => {
                info!("twilio websocket closed");
                break;
            }
            _ => continue,
        };

        let value: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                warn!("bad Twilio JSON: {}", e);
                continue;
            }
        };

        let event = value
            .get("event")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        match event {
            "connected" => {
                debug!("twilio connected event: {}", value);
            }

            "start" => {
                let sid = value
                    .get("streamSid")
                    .and_then(|v| v.as_str())
                    .or_else(|| {
                        value
                            .get("start")
                            .and_then(|s| s.get("streamSid"))
                            .and_then(|v| v.as_str())
                    })
                    .ok_or_else(|| anyhow!("missing streamSid"))?
                    .to_string();

                let call_sid = value
                    .get("start")
                    .and_then(|s| s.get("callSid"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                call_tracker = Some(CallTracker::start(
                    state.call_tracking.clone(),
                    CallStart {
                        call_sid: call_sid.clone(),
                        stream_sid: sid.clone(),
                        persona: state.persona_name.clone(),
                        scam_type: state.scam_type.clone(),
                    },
                ));

                info!("twilio start streamSid={} callSid={}", sid, call_sid);
                info!("starting call pipeline: {:?}", state.orchestrator_stack);

                pipeline = Some(match state.orchestrator_stack {
                    OrchestratorStack::OpenAiRealtime => {
                        let oai = connect_openai_realtime(
                            state.clone(),
                            sid.clone(),
                            twilio_out_tx.clone(),
                            realtime_signals.clone(),
                            call_tracker.clone(),
                        )
                        .await?;

                        let (audio_frame_tx, audio_frame_rx) =
                            mpsc::unbounded_channel::<AudioFrame>();
                        let (interrupt_frame_tx, interrupt_frame_rx) =
                            mpsc::unbounded_channel::<Vec<u8>>();

                        spawn_audio_forwarder(oai.clone(), audio_frame_rx);
                        spawn_interrupt_detector(
                            oai.clone(),
                            sid.clone(),
                            twilio_out_tx.clone(),
                            realtime_signals.clone(),
                            state.turn.clone(),
                            interrupt_frame_rx,
                        );

                        CallPipeline::OpenAiRealtime {
                            audio_tx: audio_frame_tx,
                            interrupt_tx: interrupt_frame_tx,
                        }
                    }
                    OrchestratorStack::SplitProviders => {
                        let split = connect_split_providers(
                            state.clone(),
                            sid.clone(),
                            twilio_out_tx.clone(),
                            realtime_signals.clone(),
                            call_tracker.clone(),
                        )
                        .await?;
                        let (interrupt_frame_tx, interrupt_frame_rx) =
                            mpsc::unbounded_channel::<Vec<u8>>();

                        spawn_split_interrupt_detector(
                            split.control_tx.clone(),
                            sid.clone(),
                            twilio_out_tx.clone(),
                            realtime_signals.clone(),
                            state.turn.clone(),
                            interrupt_frame_rx,
                        );

                        CallPipeline::SplitProviders {
                            audio_tx: split.audio_tx,
                            interrupt_tx: interrupt_frame_tx,
                        }
                    }
                });
            }

            "media" => {
                if pipeline.is_none() {
                    continue;
                }

                let Some(payload) = value
                    .get("media")
                    .and_then(|m| m.get("payload"))
                    .and_then(|v| v.as_str())
                else {
                    continue;
                };

                let decoded = match general_purpose::STANDARD.decode(payload) {
                    Ok(decoded) => decoded,
                    Err(e) => {
                        warn!("bad Twilio media payload: {}", e);
                        continue;
                    }
                };

                if let Some(pipeline) = pipeline.as_ref() {
                    match pipeline {
                        CallPipeline::OpenAiRealtime {
                            audio_tx,
                            interrupt_tx,
                            ..
                        } => {
                            if let Err(e) = audio_tx.send(AudioFrame {
                                b64_payload: payload.to_string(),
                            }) {
                                warn!(
                                    "failed sending Twilio audio frame to OpenAI forwarder: {}",
                                    e
                                );
                            }

                            if let Err(e) = interrupt_tx.send(decoded) {
                                warn!("failed sending frame to interrupt detector: {}", e);
                            }
                        }
                        CallPipeline::SplitProviders {
                            audio_tx,
                            interrupt_tx,
                            ..
                        } => {
                            if let Err(e) = audio_tx.send(decoded.clone()) {
                                warn!(
                                    "failed sending Twilio audio frame to Deepgram forwarder: {}",
                                    e
                                );
                            }

                            if let Err(e) = interrupt_tx.send(decoded) {
                                warn!("failed sending frame to split interrupt detector: {}", e);
                            }
                        }
                    }
                }
            }

            "mark" => {
                let mark_name = value
                    .get("mark")
                    .and_then(|m| m.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                if let Some((generation, audio_ms)) = parse_playback_mark(mark_name) {
                    if generation == response_generation.load(Ordering::Relaxed) {
                        played_audio_ms.fetch_add(audio_ms, Ordering::Relaxed);
                        queued_audio_ms
                            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                                Some(current.saturating_sub(audio_ms))
                            })
                            .ok();
                    }
                }

                let remaining = pending_playback_marks
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                        current.checked_sub(1)
                    })
                    .unwrap_or(0)
                    .saturating_sub(1);

                if remaining == 0 {
                    assistant_speaking.store(false, Ordering::Relaxed);
                }
            }

            "stop" => {
                info!(
                    "twilio stop after {:?}",
                    Instant::now().duration_since(call_started_at)
                );
                break;
            }

            other => {
                debug!("unhandled Twilio event {}: {}", other, value);
            }
        }
    }

    info!("twilio session ended");

    if let Some(tracker) = call_tracker {
        tracker.finish(FinalCallMetrics {
            generated_audio_ms: generated_audio_ms.load(Ordering::Relaxed),
            played_audio_ms: played_audio_ms.load(Ordering::Relaxed),
            queued_audio_ms: queued_audio_ms.load(Ordering::Relaxed),
            pending_marks: pending_playback_marks.load(Ordering::Relaxed),
            humanization_json: humanization_metrics(&state.humanization),
        });
    }

    Ok(())
}

fn spawn_audio_forwarder(openai: OpenAiHandle, mut audio_rx: mpsc::UnboundedReceiver<AudioFrame>) {
    tokio::spawn(async move {
        while let Some(frame) = audio_rx.recv().await {
            openai.append_audio_ulaw(&frame.b64_payload).await;
        }
    });
}

fn spawn_interrupt_detector(
    openai: OpenAiHandle,
    stream_sid: String,
    twilio_out_tx: mpsc::Sender<Value>,
    signals: RealtimeSignals,
    turn: TurnTakingConfig,
    mut interrupt_rx: mpsc::UnboundedReceiver<Vec<u8>>,
) {
    let (barge_tx, mut barge_rx) = mpsc::unbounded_channel::<u32>();
    let detector_signals = signals.clone();

    tokio::task::spawn_blocking(move || {
        let mut vad = WebRtcBargeInVad::new();
        let mut barge_in_speech_ms: u32 = 0;

        while let Some(decoded) = interrupt_rx.blocking_recv() {
            let speech = vad.is_speech_ulaw_bytes(&decoded);
            let playback_active = detector_signals.assistant_speaking.load(Ordering::Relaxed)
                || detector_signals
                    .pending_playback_marks
                    .load(Ordering::Relaxed)
                    > 0;
            let enough_assistant_audio =
                detector_signals.generated_audio_ms.load(Ordering::Relaxed)
                    >= turn.min_assistant_audio_before_barge_ms;

            if playback_active && enough_assistant_audio && speech {
                barge_in_speech_ms += MEDIA_FRAME_MS;
            } else {
                barge_in_speech_ms = 0;
                detector_signals.interrupted.store(false, Ordering::Relaxed);
            }

            if playback_active
                && enough_assistant_audio
                && barge_in_speech_ms >= turn.barge_in_speech_ms
                && !detector_signals.interrupted.swap(true, Ordering::Relaxed)
            {
                if barge_tx.send(barge_in_speech_ms).is_err() {
                    break;
                }
                barge_in_speech_ms = 0;
            }
        }
    });

    tokio::spawn(async move {
        while let Some(barge_in_speech_ms) = barge_rx.recv().await {
            info!("barge-in detected after {}ms speech", barge_in_speech_ms);
            info!(
                "playback_metrics barge_in generated_audio={}ms played_audio={}ms queued_audio={}ms pending_marks={}",
                signals.generated_audio_ms.load(Ordering::Relaxed),
                signals.played_audio_ms.load(Ordering::Relaxed),
                signals.queued_audio_ms.load(Ordering::Relaxed),
                signals.pending_playback_marks.load(Ordering::Relaxed)
            );

            signals.response_generation.fetch_add(1, Ordering::Relaxed);
            signals.generated_audio_ms.store(0, Ordering::Relaxed);
            signals.played_audio_ms.store(0, Ordering::Relaxed);
            signals.queued_audio_ms.store(0, Ordering::Relaxed);
            signals.pending_playback_marks.store(0, Ordering::Relaxed);
            signals.assistant_speaking.store(false, Ordering::Relaxed);
            *signals.active_response_id.lock().await = None;
            *signals.response_requested_at.lock().await = None;

            let clear_msg = json!({
                "event": "clear",
                "streamSid": stream_sid
            });

            if let Err(e) = twilio_out_tx.send(clear_msg).await {
                warn!("failed queueing immediate Twilio clear: {}", e);
            }

            if signals.response_active.load(Ordering::Relaxed) {
                openai.cancel_response().await;
            }
        }
    });
}

fn spawn_split_interrupt_detector(
    control_tx: mpsc::UnboundedSender<SplitControl>,
    stream_sid: String,
    twilio_out_tx: mpsc::Sender<Value>,
    signals: RealtimeSignals,
    turn: TurnTakingConfig,
    mut interrupt_rx: mpsc::UnboundedReceiver<Vec<u8>>,
) {
    let (barge_tx, mut barge_rx) = mpsc::unbounded_channel::<u32>();
    let detector_signals = signals.clone();

    tokio::task::spawn_blocking(move || {
        let mut vad = WebRtcBargeInVad::new();
        let mut barge_in_speech_ms: u32 = 0;

        while let Some(decoded) = interrupt_rx.blocking_recv() {
            let speech = vad.is_speech_ulaw_bytes(&decoded);
            let playback_active = detector_signals.assistant_speaking.load(Ordering::Relaxed)
                || detector_signals
                    .pending_playback_marks
                    .load(Ordering::Relaxed)
                    > 0;
            let enough_assistant_audio =
                detector_signals.generated_audio_ms.load(Ordering::Relaxed)
                    >= turn.min_assistant_audio_before_barge_ms;

            if playback_active && enough_assistant_audio && speech {
                barge_in_speech_ms += MEDIA_FRAME_MS;
            } else {
                barge_in_speech_ms = 0;
                detector_signals.interrupted.store(false, Ordering::Relaxed);
            }

            if playback_active
                && enough_assistant_audio
                && barge_in_speech_ms >= turn.barge_in_speech_ms
                && !detector_signals.interrupted.swap(true, Ordering::Relaxed)
            {
                if barge_tx.send(barge_in_speech_ms).is_err() {
                    break;
                }
                barge_in_speech_ms = 0;
            }
        }
    });

    tokio::spawn(async move {
        while let Some(barge_in_speech_ms) = barge_rx.recv().await {
            signals.response_generation.fetch_add(1, Ordering::Relaxed);
            info!(
                "split barge-in detected after {}ms speech",
                barge_in_speech_ms
            );
            info!(
                "playback_metrics split_barge_in generated_audio={}ms played_audio={}ms queued_audio={}ms pending_marks={}",
                signals.generated_audio_ms.load(Ordering::Relaxed),
                signals.played_audio_ms.load(Ordering::Relaxed),
                signals.queued_audio_ms.load(Ordering::Relaxed),
                signals.pending_playback_marks.load(Ordering::Relaxed)
            );

            signals.generated_audio_ms.store(0, Ordering::Relaxed);
            signals.played_audio_ms.store(0, Ordering::Relaxed);
            signals.queued_audio_ms.store(0, Ordering::Relaxed);
            signals.pending_playback_marks.store(0, Ordering::Relaxed);
            signals.assistant_speaking.store(false, Ordering::Relaxed);
            signals.response_active.store(false, Ordering::Relaxed);

            let clear_msg = json!({
                "event": "clear",
                "streamSid": stream_sid
            });

            if let Err(e) = twilio_out_tx.send(clear_msg).await {
                warn!("failed queueing split Twilio clear: {}", e);
            }

            if let Err(e) = control_tx.send(SplitControl::Cancel { context_id: None }) {
                warn!("failed sending split cancel: {}", e);
            }
        }
    });
}
