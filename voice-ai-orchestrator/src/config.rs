use crate::humanize::HumanizationConfig;
use crate::tracking::{repository_from_path, SharedCallTrackingRepository};
use anyhow::{Context, Result};
use std::env;
pub(crate) const DEFAULT_DEEPGRAM_MODEL: &str = "nova-3";
pub(crate) const DEFAULT_DEEPGRAM_LANGUAGE: &str = "en-US";
pub(crate) const DEFAULT_DEEPGRAM_ENDPOINTING_MS: u32 = 180;
pub(crate) const DEFAULT_OPENAI_TEXT_MODEL: &str = "gpt-5.4-nano";
pub(crate) const DEFAULT_CARTESIA_MODEL: &str = "sonic-3";
pub(crate) const DEFAULT_CARTESIA_LANGUAGE: &str = "en";
pub(crate) const DEFAULT_MAX_RESPONSE_TOKENS: u32 = 64;
pub(crate) const DEFAULT_SOFT_AUDIO_LIMIT_MS: u64 = 2_500;
pub(crate) const DEFAULT_HARD_AUDIO_LIMIT_MS: u64 = 0;
pub(crate) const DEFAULT_BACKLOG_LOW_WATER_MS: u64 = 1_500;
pub(crate) const DEFAULT_BACKLOG_HIGH_WATER_MS: u64 = 2_500;

const DEFAULT_BARGE_IN_SPEECH_MS: u32 = 1_200;
const DEFAULT_MIN_ASSISTANT_AUDIO_BEFORE_BARGE_MS: u64 = 5_000;
const DEFAULT_OPENAI_VAD_THRESHOLD: f64 = 0.48;
const DEFAULT_OPENAI_VAD_PREFIX_PADDING_MS: u32 = 180;
const DEFAULT_OPENAI_VAD_SILENCE_DURATION_MS: u32 = 280;
const DEFAULT_SHORT_TURN_GRACE_MS: u64 = 260;
const DEFAULT_NORMAL_TURN_GRACE_MS: u64 = 120;
const DEFAULT_LONG_TURN_GRACE_MS: u64 = 60;
const DEFAULT_MIN_RESPONSE_SPEECH_MS: u64 = 500;

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) call_tracking: SharedCallTrackingRepository,
    pub(crate) orchestrator_stack: OrchestratorStack,
    pub(crate) openai_api_key: String,
    pub(crate) openai_model: String,
    pub(crate) openai_text_model: String,
    pub(crate) openai_voice: String,
    pub(crate) deepgram_api_key: Option<String>,
    pub(crate) deepgram_model: String,
    pub(crate) deepgram_language: String,
    pub(crate) deepgram_endpointing_ms: u32,
    pub(crate) cartesia_api_key: Option<String>,
    pub(crate) cartesia_model: String,
    pub(crate) cartesia_voice_id: Option<String>,
    pub(crate) cartesia_language: String,
    pub(crate) public_ws_url: String,
    pub(crate) persona_name: String,
    pub(crate) scam_type: String,
    pub(crate) persona_instructions: String,
    pub(crate) response_instructions: String,
    pub(crate) max_response_tokens: u32,
    pub(crate) soft_audio_limit_ms: u64,
    pub(crate) hard_audio_limit_ms: u64,
    pub(crate) backlog_low_water_ms: u64,
    pub(crate) backlog_high_water_ms: u64,
    pub(crate) turn: TurnTakingConfig,
    pub(crate) humanization: HumanizationConfig,
}

impl AppState {
    pub(crate) fn from_env() -> Result<Self> {
        let orchestrator_stack = OrchestratorStack::from_env();

        Ok(Self {
            call_tracking: repository_from_path(
                env::var("CALL_TRACKING_JSONL_PATH")
                    .unwrap_or_else(|_| "data/call_tracking.jsonl".to_string()),
            ),
            orchestrator_stack,
            openai_api_key: env::var("OPENAI_API_KEY").context("OPENAI_API_KEY missing")?,
            openai_model: env::var("OPENAI_REALTIME_MODEL")
                .unwrap_or_else(|_| "gpt-realtime".to_string()),
            openai_text_model: env::var("OPENAI_TEXT_MODEL")
                .unwrap_or_else(|_| DEFAULT_OPENAI_TEXT_MODEL.to_string()),
            openai_voice: env::var("OPENAI_VOICE").unwrap_or_else(|_| "alloy".to_string()),
            deepgram_api_key: env_nonempty("DEEPGRAM_API_KEY"),
            deepgram_model: env::var("DEEPGRAM_MODEL")
                .unwrap_or_else(|_| DEFAULT_DEEPGRAM_MODEL.to_string()),
            deepgram_language: env::var("DEEPGRAM_LANGUAGE")
                .unwrap_or_else(|_| DEFAULT_DEEPGRAM_LANGUAGE.to_string()),
            deepgram_endpointing_ms: env_u32(
                "DEEPGRAM_ENDPOINTING_MS",
                DEFAULT_DEEPGRAM_ENDPOINTING_MS,
            ),
            cartesia_api_key: env_nonempty("CARTESIA_API_KEY"),
            cartesia_model: env::var("CARTESIA_MODEL")
                .unwrap_or_else(|_| DEFAULT_CARTESIA_MODEL.to_string()),
            cartesia_voice_id: env_nonempty("CARTESIA_VOICE_ID"),
            cartesia_language: env::var("CARTESIA_LANGUAGE")
                .unwrap_or_else(|_| DEFAULT_CARTESIA_LANGUAGE.to_string()),
            public_ws_url: env::var("PUBLIC_WS_URL")
                .unwrap_or_else(|_| "wss://example.ngrok-free.app/twilio/media".to_string()),
            persona_name: env::var("PERSONA_NAME").unwrap_or_else(|_| "default".to_string()),
            scam_type: env::var("SCAM_TYPE").unwrap_or_else(|_| "unknown".to_string()),
            persona_instructions: env::var("PERSONA_INSTRUCTIONS").unwrap_or_else(|_| {
                r#"
You are a natural, human-like phone persona. You are polite, slightly confused, and conversational.
Your goal is to keep suspicious callers talking while safely extracting information such as:
phone numbers, names, callback numbers, websites, crypto wallet addresses, payment instructions,
company names, and scripts.

Never provide real private information.
Never make real payments.
Ask clarifying questions naturally.
Keep responses short and interruptible.
If interrupted, stop talking and listen.
Prefer very short spoken replies. In normal conversation, speak for under 2 seconds.
Use one concise sentence unless the caller explicitly asks for details.
"#
                .to_string()
            }),
            response_instructions: env::var("RESPONSE_INSTRUCTIONS").unwrap_or_else(|_| {
                "Keep every reply extremely short. Use one sentence whenever possible. Prefer acknowledgements and direct questions over explanations. Never monologue. Avoid compound sentences, lists, and overexplaining. Yield the floor frequently."
                    .to_string()
            }),
            max_response_tokens: env_u32("MAX_RESPONSE_TOKENS", DEFAULT_MAX_RESPONSE_TOKENS),
            soft_audio_limit_ms: env_u64("SOFT_AUDIO_LIMIT_MS", DEFAULT_SOFT_AUDIO_LIMIT_MS),
            hard_audio_limit_ms: env_u64("HARD_AUDIO_LIMIT_MS", DEFAULT_HARD_AUDIO_LIMIT_MS),
            backlog_low_water_ms: env_u64("BACKLOG_LOW_WATER_MS", DEFAULT_BACKLOG_LOW_WATER_MS),
            backlog_high_water_ms: env_u64("BACKLOG_HIGH_WATER_MS", DEFAULT_BACKLOG_HIGH_WATER_MS),
            turn: TurnTakingConfig::from_env(),
            humanization: HumanizationConfig::from_env(),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OrchestratorStack {
    OpenAiRealtime,
    SplitProviders,
}

impl OrchestratorStack {
    pub(crate) fn from_env() -> Self {
        match env::var("ORCHESTRATOR_STACK")
            .unwrap_or_else(|_| "openai_realtime".to_string())
            .to_lowercase()
            .as_str()
        {
            "split" | "split_providers" | "deepgram_openai_cartesia" => Self::SplitProviders,
            _ => Self::OpenAiRealtime,
        }
    }
}

#[derive(Clone)]
pub(crate) struct TurnTakingConfig {
    pub(crate) barge_in_speech_ms: u32,
    pub(crate) min_assistant_audio_before_barge_ms: u64,
    pub(crate) vad_threshold: f64,
    pub(crate) vad_prefix_padding_ms: u32,
    pub(crate) vad_silence_duration_ms: u32,
    pub(crate) short_turn_grace_ms: u64,
    pub(crate) normal_turn_grace_ms: u64,
    pub(crate) long_turn_grace_ms: u64,
    pub(crate) min_response_speech_ms: u64,
}

impl TurnTakingConfig {
    pub(crate) fn from_env() -> Self {
        Self {
            barge_in_speech_ms: env_u32("BARGE_IN_SPEECH_MS", DEFAULT_BARGE_IN_SPEECH_MS),
            min_assistant_audio_before_barge_ms: env_u64(
                "MIN_ASSISTANT_AUDIO_BEFORE_BARGE_MS",
                DEFAULT_MIN_ASSISTANT_AUDIO_BEFORE_BARGE_MS,
            ),
            vad_threshold: env_f64("OPENAI_VAD_THRESHOLD", DEFAULT_OPENAI_VAD_THRESHOLD),
            vad_prefix_padding_ms: env_u32(
                "OPENAI_VAD_PREFIX_PADDING_MS",
                DEFAULT_OPENAI_VAD_PREFIX_PADDING_MS,
            ),
            vad_silence_duration_ms: env_u32(
                "OPENAI_VAD_SILENCE_DURATION_MS",
                DEFAULT_OPENAI_VAD_SILENCE_DURATION_MS,
            ),
            short_turn_grace_ms: env_u64("SHORT_TURN_GRACE_MS", DEFAULT_SHORT_TURN_GRACE_MS),
            normal_turn_grace_ms: env_u64("NORMAL_TURN_GRACE_MS", DEFAULT_NORMAL_TURN_GRACE_MS),
            long_turn_grace_ms: env_u64("LONG_TURN_GRACE_MS", DEFAULT_LONG_TURN_GRACE_MS),
            min_response_speech_ms: env_u64(
                "MIN_RESPONSE_SPEECH_MS",
                DEFAULT_MIN_RESPONSE_SPEECH_MS,
            ),
        }
    }
}

pub(crate) fn env_u32(name: &str, default: u32) -> u32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

pub(crate) fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

pub(crate) fn env_f64(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

pub(crate) fn env_nonempty(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.trim().is_empty())
}
