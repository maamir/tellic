use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tracing::warn;

pub(crate) trait CallTrackingRepository: Send + Sync {
    fn save_call(&self, record: &CallRecord) -> Result<()>;
    fn save_intelligence(&self, record: &CallIntelligence) -> Result<()>;
}

pub(crate) type SharedCallTrackingRepository = Arc<dyn CallTrackingRepository>;

pub(crate) struct JsonlCallTrackingRepository {
    path: PathBuf,
    lock: Mutex<()>,
}

impl JsonlCallTrackingRepository {
    pub(crate) fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            lock: Mutex::new(()),
        }
    }

    fn append(&self, kind: &str, value: Value) -> Result<()> {
        let _guard = self.lock.lock().expect("tracking lock poisoned");
        if let Some(parent) = self
            .path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed creating tracking dir {:?}", parent))?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("failed opening tracking file {:?}", self.path))?;
        let line = json!({
            "type": kind,
            "record": value,
        });
        writeln!(file, "{line}").context("failed writing tracking record")?;
        Ok(())
    }
}

impl CallTrackingRepository for JsonlCallTrackingRepository {
    fn save_call(&self, record: &CallRecord) -> Result<()> {
        self.append("call", serde_json::to_value(record)?)
    }

    fn save_intelligence(&self, record: &CallIntelligence) -> Result<()> {
        self.append("call_intelligence", serde_json::to_value(record)?)
    }
}

#[derive(Clone)]
pub(crate) struct CallTracker {
    repository: SharedCallTrackingRepository,
    state: Arc<Mutex<CallTrackingState>>,
}

#[derive(Clone)]
pub(crate) struct CallStart {
    pub(crate) call_sid: String,
    pub(crate) stream_sid: String,
    pub(crate) persona: String,
    pub(crate) scam_type: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct TranscriptEntry {
    pub(crate) role: String,
    pub(crate) text: String,
    pub(crate) at_ms: u128,
    pub(crate) created_at: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct CallRecord {
    pub(crate) call_sid: String,
    pub(crate) stream_sid: String,
    pub(crate) persona: String,
    pub(crate) scam_type: String,
    pub(crate) start_time: String,
    pub(crate) end_time: String,
    pub(crate) duration_seconds: u64,
    pub(crate) transcript_text: String,
    pub(crate) transcript_json: Value,
    pub(crate) live_entities_json: Value,
    pub(crate) metrics_json: Value,
    pub(crate) created_at: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct CallIntelligence {
    pub(crate) call_sid: String,
    pub(crate) scam_type: String,
    pub(crate) impersonated_organisation: Option<String>,
    pub(crate) summary: String,
    pub(crate) confidence: f64,
    pub(crate) extraction_json: Value,
    pub(crate) created_at: String,
}

pub(crate) struct FinalCallMetrics {
    pub(crate) generated_audio_ms: u64,
    pub(crate) played_audio_ms: u64,
    pub(crate) queued_audio_ms: u64,
    pub(crate) pending_marks: usize,
    pub(crate) humanization_json: Value,
}

struct CallTrackingState {
    call_sid: String,
    stream_sid: String,
    persona: String,
    scam_type: String,
    start_time: String,
    created_at: String,
    started_at: Instant,
    transcript: Vec<TranscriptEntry>,
    live_entities_json: Value,
}

impl CallTracker {
    pub(crate) fn start(repository: SharedCallTrackingRepository, start: CallStart) -> Self {
        let created_at = now_timestamp();
        Self {
            repository,
            state: Arc::new(Mutex::new(CallTrackingState {
                call_sid: start.call_sid,
                stream_sid: start.stream_sid,
                persona: start.persona,
                scam_type: start.scam_type,
                start_time: created_at.clone(),
                created_at,
                started_at: Instant::now(),
                transcript: Vec::new(),
                live_entities_json: json!({}),
            })),
        }
    }

    pub(crate) fn record_user_text(&self, text: impl Into<String>) {
        self.record_transcript("user", text);
    }

    pub(crate) fn record_assistant_text(&self, text: impl Into<String>) {
        self.record_transcript("assistant", text);
    }

    fn record_transcript(&self, role: &str, text: impl Into<String>) {
        let text = text.into();
        if text.trim().is_empty() {
            return;
        }

        let mut state = self.state.lock().expect("call tracking lock poisoned");
        let at_ms = state.started_at.elapsed().as_millis();
        state.transcript.push(TranscriptEntry {
            role: role.to_string(),
            text,
            at_ms,
            created_at: now_timestamp(),
        });
        state.live_entities_json = extract_entities(&state.transcript_text());
    }

    pub(crate) fn finish(&self, metrics: FinalCallMetrics) {
        let (call, intelligence) = {
            let state = self.state.lock().expect("call tracking lock poisoned");
            let transcript_text = state.transcript_text();
            let extraction_json = extract_entities(&transcript_text);
            let intelligence =
                build_intelligence(&state.call_sid, &state.scam_type, &transcript_text);

            (
                CallRecord {
                    call_sid: state.call_sid.clone(),
                    stream_sid: state.stream_sid.clone(),
                    persona: state.persona.clone(),
                    scam_type: state.scam_type.clone(),
                    start_time: state.start_time.clone(),
                    end_time: now_timestamp(),
                    duration_seconds: elapsed_seconds(state.started_at.elapsed()),
                    transcript_text,
                    transcript_json: json!(state.transcript),
                    live_entities_json: extraction_json.clone(),
                    metrics_json: json!({
                        "generated_audio_ms": metrics.generated_audio_ms,
                        "played_audio_ms": metrics.played_audio_ms,
                        "queued_audio_ms": metrics.queued_audio_ms,
                        "pending_marks": metrics.pending_marks,
                        "humanization": metrics.humanization_json,
                    }),
                    created_at: state.created_at.clone(),
                },
                CallIntelligence {
                    call_sid: state.call_sid.clone(),
                    scam_type: state.scam_type.clone(),
                    impersonated_organisation: intelligence.impersonated_organisation,
                    summary: intelligence.summary,
                    confidence: intelligence.confidence,
                    extraction_json,
                    created_at: now_timestamp(),
                },
            )
        };

        if let Err(e) = self.repository.save_call(&call) {
            warn!("failed saving call tracking record: {:?}", e);
        }
        if let Err(e) = self.repository.save_intelligence(&intelligence) {
            warn!("failed saving call intelligence record: {:?}", e);
        }
    }
}

impl CallTrackingState {
    fn transcript_text(&self) -> String {
        self.transcript
            .iter()
            .map(|entry| format!("{}: {}", entry.role, entry.text))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

struct IntelligenceDraft {
    impersonated_organisation: Option<String>,
    summary: String,
    confidence: f64,
}

fn build_intelligence(call_sid: &str, scam_type: &str, transcript_text: &str) -> IntelligenceDraft {
    let entities = extract_entities(transcript_text);
    let impersonated_organisation = entities
        .get("organisations")
        .and_then(|value| value.as_array())
        .and_then(|items| items.first())
        .and_then(|value| value.as_str())
        .map(str::to_string);

    let summary = if transcript_text.trim().is_empty() {
        format!("Call {call_sid} ended before a transcript was captured.")
    } else {
        let first_lines = transcript_text
            .lines()
            .take(4)
            .collect::<Vec<_>>()
            .join(" ");
        format!("Observed {scam_type} call. {first_lines}")
    };

    let confidence = if impersonated_organisation.is_some() || transcript_text.len() > 120 {
        0.72
    } else if transcript_text.trim().is_empty() {
        0.1
    } else {
        0.45
    };

    IntelligenceDraft {
        impersonated_organisation,
        summary,
        confidence,
    }
}

fn extract_entities(text: &str) -> Value {
    let phones = extract_phone_like_values(text);
    let urls = text
        .split_whitespace()
        .filter_map(normalize_url_like)
        .collect::<Vec<_>>();
    let emails = text
        .split_whitespace()
        .filter_map(normalize_email_like)
        .collect::<Vec<_>>();
    let organisations = known_organisations(text);

    json!({
        "phones": unique(phones),
        "urls": unique(urls),
        "emails": unique(emails),
        "organisations": unique(organisations),
    })
}

fn extract_phone_like_values(text: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut current = String::new();
    let mut digit_count = 0;

    for ch in text.chars().chain(std::iter::once(' ')) {
        if ch.is_ascii_digit() || matches!(ch, '+' | '(' | ')' | '-' | ' ') {
            if ch.is_ascii_digit() {
                digit_count += 1;
            }
            current.push(ch);
        } else {
            if digit_count >= 7 {
                values.push(current.trim().to_string());
            }
            current.clear();
            digit_count = 0;
        }
    }

    values
}

fn normalize_url_like(token: &str) -> Option<String> {
    let token = token.trim_matches(|ch: char| matches!(ch, ',' | '.' | ';' | ')' | '(' | '"'));
    let lower = token.to_ascii_lowercase();
    if lower.starts_with("http://")
        || lower.starts_with("https://")
        || lower.starts_with("www.")
        || (lower.contains('.') && !lower.contains('@') && lower.len() > 4)
    {
        Some(token.to_string())
    } else {
        None
    }
}

fn normalize_email_like(token: &str) -> Option<String> {
    let token = token.trim_matches(|ch: char| matches!(ch, ',' | '.' | ';' | ')' | '(' | '"'));
    if token.contains('@') && token.contains('.') {
        Some(token.to_string())
    } else {
        None
    }
}

fn known_organisations(text: &str) -> Vec<String> {
    let lower = text.to_ascii_lowercase();
    [
        ("amazon", "Amazon"),
        ("apple", "Apple"),
        ("bank of america", "Bank of America"),
        ("coinbase", "Coinbase"),
        ("irs", "IRS"),
        ("microsoft", "Microsoft"),
        ("paypal", "PayPal"),
        ("social security", "Social Security Administration"),
        ("wells fargo", "Wells Fargo"),
        ("zelle", "Zelle"),
    ]
    .into_iter()
    .filter_map(|(needle, label)| lower.contains(needle).then_some(label.to_string()))
    .collect()
}

fn unique(values: Vec<String>) -> Vec<String> {
    values.into_iter().fold(Vec::new(), |mut acc, value| {
        if !acc.iter().any(|existing| existing == &value) {
            acc.push(value);
        }
        acc
    })
}

fn now_timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}.{}", now.as_secs(), now.subsec_millis())
}

fn elapsed_seconds(duration: Duration) -> u64 {
    duration.as_secs().max(1)
}

pub(crate) fn repository_from_path(path: impl AsRef<Path>) -> SharedCallTrackingRepository {
    Arc::new(JsonlCallTrackingRepository::new(path.as_ref()))
}
