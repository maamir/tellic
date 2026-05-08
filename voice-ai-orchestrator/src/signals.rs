use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize},
    Arc,
};
use std::time::Instant;
use tokio::sync::Mutex;

#[derive(Clone)]
pub(crate) struct RealtimeSignals {
    pub(crate) assistant_speaking: Arc<AtomicBool>,
    pub(crate) response_active: Arc<AtomicBool>,
    pub(crate) interrupted: Arc<AtomicBool>,
    pub(crate) pending_playback_marks: Arc<AtomicUsize>,
    pub(crate) response_generation: Arc<AtomicU64>,
    pub(crate) generated_audio_ms: Arc<AtomicU64>,
    pub(crate) played_audio_ms: Arc<AtomicU64>,
    pub(crate) queued_audio_ms: Arc<AtomicU64>,
    pub(crate) active_response_id: Arc<Mutex<Option<String>>>,
    pub(crate) response_requested_at: Arc<Mutex<Option<(u64, Instant)>>>,
    pub(crate) user_speech_generation: Arc<AtomicU64>,
    pub(crate) current_turn_speech_ms: Arc<AtomicU64>,
    pub(crate) openai_speech_started_ms: Arc<AtomicU64>,
}
