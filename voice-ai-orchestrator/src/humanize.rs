use crate::audio::ULAW_SAMPLE_RATE;
use serde_json::{json, Value};
use std::{env, fs, path::Path};
use tracing::warn;

const SAMPLE_RATE: u32 = ULAW_SAMPLE_RATE;
const SPEAKING_BG_GAIN: f32 = 0.05;
const SILENCE_BG_GAIN: f32 = 0.15;
const OFFICE_FX_GAIN: f32 = 0.032;
const BREATH_GAIN: f32 = 0.08;
const VOICE_HEADROOM_GAIN: f32 = 0.94;

#[derive(Clone)]
pub(crate) struct HumanizationConfig {
    pub(crate) enabled: bool,
    pub(crate) background_profile: BackgroundProfile,
    pub(crate) ambience_paths: Vec<String>,
    pub(crate) breathing_paths: Vec<String>,
    pub(crate) office_fx_paths: Vec<String>,
    pub(crate) tone: ToneProfile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BackgroundProfile {
    None,
    Office,
    CoffeeCafe,
    Kitchen,
    CleaningVacuum,
    Custom,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ToneProfile {
    Support,
    Sales,
    Complaint,
    Escalation,
}

pub(crate) struct HumanizedSpeech {
    pub(crate) text: String,
    pub(crate) tone: ToneProfile,
    pub(crate) initial_pause_ms: u64,
    pub(crate) breath_after_done: bool,
}

pub(crate) struct AudioHumanizer {
    config: HumanizationConfig,
    rng: SmallRng,
    ambience: Vec<LoopingLayer>,
    breaths: Vec<Vec<i16>>,
    office_fx: Vec<Vec<i16>>,
    active_layers: Vec<ActiveLayer>,
    samples_until_fx: usize,
    pending_silence_samples: usize,
    breath_after_done: bool,
    tone: ToneProfile,
}

struct LoopingLayer {
    samples: Vec<i16>,
    cursor: usize,
    gain: f32,
}

struct ActiveLayer {
    samples: Vec<i16>,
    cursor: usize,
    gain: f32,
}

impl HumanizationConfig {
    pub(crate) fn from_env() -> Self {
        let background_profile = BackgroundProfile::from_env(
            env::var("HUMANIZE_BACKGROUND_PROFILE")
                .unwrap_or_else(|_| "office".to_string())
                .as_str(),
        );

        Self {
            enabled: env_bool("HUMANIZATION_ENABLED", true),
            background_profile,
            ambience_paths: env_list("HUMANIZE_BACKGROUND_WAVS"),
            breathing_paths: env_list("HUMANIZE_BREATH_WAVS"),
            office_fx_paths: env_list("HUMANIZE_OFFICE_FX_WAVS"),
            tone: ToneProfile::from_env(
                env::var("HUMANIZE_TONE")
                    .unwrap_or_else(|_| "support".to_string())
                    .as_str(),
            ),
        }
    }
}

impl BackgroundProfile {
    fn from_env(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" | "off" | "silent" => Self::None,
            "coffee" | "coffee_cafe" | "cafe" | "café" => Self::CoffeeCafe,
            "kitchen" => Self::Kitchen,
            "cleaning" | "vacuum" | "cleaning_vacuum" => Self::CleaningVacuum,
            "custom" => Self::Custom,
            _ => Self::Office,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Office => "office",
            Self::CoffeeCafe => "coffee_cafe",
            Self::Kitchen => "kitchen",
            Self::CleaningVacuum => "cleaning_vacuum",
            Self::Custom => "custom",
        }
    }

    fn layer_gain(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::Office => 1.0,
            Self::CoffeeCafe => 0.85,
            Self::Kitchen => 0.75,
            Self::CleaningVacuum => 0.55,
            Self::Custom => 1.0,
        }
    }
}

impl ToneProfile {
    pub(crate) fn from_env(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "sales" => Self::Sales,
            "complaint" | "empathetic" => Self::Complaint,
            "escalation" | "serious" => Self::Escalation,
            _ => Self::Support,
        }
    }

    fn energy_gain(self) -> f32 {
        match self {
            Self::Support => 0.98,
            Self::Sales => 1.04,
            Self::Complaint => 0.94,
            Self::Escalation => 0.9,
        }
    }

    fn speed_pitch_factor(self) -> f32 {
        match self {
            Self::Support => 1.0,
            Self::Sales => 1.04,
            Self::Complaint => 0.96,
            Self::Escalation => 0.94,
        }
    }

    fn pause_density(self) -> f32 {
        match self {
            Self::Support => 1.0,
            Self::Sales => 0.75,
            Self::Complaint => 1.2,
            Self::Escalation => 1.35,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Support => "support",
            Self::Sales => "sales",
            Self::Complaint => "complaint",
            Self::Escalation => "escalation",
        }
    }
}

impl HumanizedSpeech {
    pub(crate) fn from_response(raw: &str, default_tone: ToneProfile) -> Self {
        let tone = detect_tone(raw).unwrap_or(default_tone);
        let mut rng = SmallRng::new(hash_seed(raw));
        let normalized = raw.split_whitespace().collect::<Vec<_>>().join(" ");
        let text = humanize_text(&normalized, tone, &mut rng);
        let punctuation = punctuation_profile(&text);
        let initial_pause_ms = if punctuation.long_response {
            rng.range_u64(500, 900)
        } else if matches!(tone, ToneProfile::Complaint | ToneProfile::Escalation) && rng.chance(12)
        {
            rng.range_u64(160, 320)
        } else {
            0
        };

        Self {
            text,
            tone,
            initial_pause_ms,
            breath_after_done: punctuation.has_sentence_end && rng.chance(7),
        }
    }

    pub(crate) fn from_spoken_text(text: &str, default_tone: ToneProfile) -> Self {
        let tone = detect_tone(text).unwrap_or(default_tone);
        let mut rng = SmallRng::new(hash_seed(text));
        let punctuation = punctuation_profile(text);
        let initial_pause_ms = if punctuation.long_response {
            rng.range_u64(500, 900)
        } else {
            0
        };

        Self {
            text: text.to_string(),
            tone,
            initial_pause_ms,
            breath_after_done: punctuation.has_sentence_end && rng.chance(7),
        }
    }
}

impl AudioHumanizer {
    pub(crate) fn new(config: HumanizationConfig, speech: &HumanizedSpeech) -> Self {
        let ambience = config
            .ambience_paths
            .iter()
            .filter_map(|path| load_wav_i16(path))
            .filter(|samples| !samples.is_empty())
            .map(|samples| LoopingLayer {
                samples,
                cursor: 0,
                gain: config.background_profile.layer_gain(),
            })
            .collect::<Vec<_>>();
        let breaths = config
            .breathing_paths
            .iter()
            .filter_map(|path| load_wav_i16(path))
            .filter(|samples| !samples.is_empty())
            .collect::<Vec<_>>();
        let office_fx = config
            .office_fx_paths
            .iter()
            .filter_map(|path| load_wav_i16(path))
            .filter(|samples| !samples.is_empty())
            .collect::<Vec<_>>();
        let mut rng = SmallRng::new(hash_seed(&speech.text));
        let samples_until_fx = rng.range_usize(1_200, 6_400);

        Self {
            config,
            rng,
            ambience,
            breaths,
            office_fx,
            active_layers: Vec::new(),
            samples_until_fx,
            pending_silence_samples: ms_to_samples(speech.initial_pause_ms),
            breath_after_done: speech.breath_after_done,
            tone: speech.tone,
        }
    }

    pub(crate) fn process_chunk(&mut self, voice_samples: &[i16], interrupted: bool) -> Vec<i16> {
        if interrupted {
            self.clear();
            return Vec::new();
        }

        if !self.config.enabled {
            return voice_samples.to_vec();
        }

        let mut output = Vec::with_capacity(self.pending_silence_samples + voice_samples.len());

        while self.pending_silence_samples > 0 {
            let take = self.pending_silence_samples.min(160);
            output.extend(self.mix_frame(&vec![0; take], false));
            self.pending_silence_samples -= take;
        }

        for frame in voice_samples.chunks(160) {
            let normalized = normalize_voice_frame(frame);
            let toned = apply_speed_pitch(&normalized, self.tone.speed_pitch_factor());
            let speaking = is_speaking(&toned);
            maybe_schedule_office_fx(self, speaking, toned.len());
            output.extend(self.mix_frame(&toned, speaking));
        }

        output
    }

    pub(crate) fn finish(&mut self) -> Vec<i16> {
        if !self.config.enabled {
            return Vec::new();
        }

        if self.breath_after_done {
            self.schedule_breath();
        }

        let mut output = Vec::new();
        while !self.active_layers.is_empty() {
            output.extend(self.mix_frame(&[0; 160], false));
            if output.len() > ms_to_samples(900) {
                break;
            }
        }
        output
    }

    pub(crate) fn clear(&mut self) {
        self.active_layers.clear();
        self.pending_silence_samples = 0;
        self.breath_after_done = false;
    }

    fn mix_frame(&mut self, voice_frame: &[i16], speaking: bool) -> Vec<i16> {
        let bg_gain = if speaking {
            SPEAKING_BG_GAIN
        } else {
            SILENCE_BG_GAIN
        };
        let voice_gain = VOICE_HEADROOM_GAIN * self.tone.energy_gain();
        let mut mixed = Vec::with_capacity(voice_frame.len());

        for sample in voice_frame {
            let mut value = *sample as f32 * voice_gain;

            for ambience in self.ambience.iter_mut() {
                value += ambience.next_sample() as f32 * bg_gain * ambience.gain;
            }

            for layer in self.active_layers.iter_mut() {
                value += layer.next_sample() as f32 * layer.gain;
            }

            mixed.push(soft_clip_i16(value));
        }

        self.active_layers.retain(|layer| !layer.done());
        mixed
    }

    fn schedule_breath(&mut self) {
        if self.breaths.is_empty() {
            return;
        }
        let index = self
            .rng
            .range_usize(0, self.breaths.len().saturating_sub(1));
        self.active_layers.push(ActiveLayer {
            samples: self.breaths[index].clone(),
            cursor: 0,
            gain: BREATH_GAIN,
        });
    }
}

impl LoopingLayer {
    fn next_sample(&mut self) -> i16 {
        if self.samples.is_empty() {
            return 0;
        }
        let sample = self.samples[self.cursor];
        self.cursor = (self.cursor + 1) % self.samples.len();
        sample
    }
}

impl ActiveLayer {
    fn next_sample(&mut self) -> i16 {
        if self.cursor >= self.samples.len() {
            return 0;
        }
        let sample = self.samples[self.cursor];
        self.cursor += 1;
        sample
    }

    fn done(&self) -> bool {
        self.cursor >= self.samples.len()
    }
}

pub(crate) fn humanization_metrics(config: &HumanizationConfig) -> Value {
    json!({
        "enabled": config.enabled,
        "tone": config.tone.label(),
        "background_profile": config.background_profile.label(),
        "speaking_bg_gain": SPEAKING_BG_GAIN,
        "silence_bg_gain": SILENCE_BG_GAIN,
        "ambience_assets": config.ambience_paths.len(),
        "breath_assets": config.breathing_paths.len(),
        "office_fx_assets": config.office_fx_paths.len(),
        "ambience_configured": !config.ambience_paths.is_empty(),
    })
}

fn humanize_text(raw: &str, tone: ToneProfile, rng: &mut SmallRng) -> String {
    if raw.is_empty() {
        return raw.to_string();
    }

    let mut text = raw.to_string();
    if should_add_ack(&text) {
        let ack = match tone {
            ToneProfile::Sales => ["Great", "Perfect", "Absolutely"][rng.range_usize(0, 2)],
            ToneProfile::Complaint => ["I understand", "I hear you", "Okay"][rng.range_usize(0, 2)],
            ToneProfile::Escalation => ["Understood", "Okay", "Right"][rng.range_usize(0, 2)],
            ToneProfile::Support => ["Sure", "Okay", "Perfect"][rng.range_usize(0, 2)],
        };
        text = format!("{ack}, {text}");
    }

    if rng.chance(6) && text.split_whitespace().count() > 6 {
        let filler = match tone {
            ToneProfile::Sales => "actually",
            ToneProfile::Complaint => "just",
            ToneProfile::Escalation => "let me be clear",
            ToneProfile::Support => "let me just",
        };
        text = insert_after_first_word(&text, filler);
    }

    text
}

fn should_add_ack(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    !(lower.starts_with("okay")
        || lower.starts_with("sure")
        || lower.starts_with("perfect")
        || lower.starts_with("understood")
        || lower.starts_with("i understand"))
}

fn insert_after_first_word(text: &str, filler: &str) -> String {
    let mut parts = text.splitn(2, ' ');
    let first = parts.next().unwrap_or_default();
    let rest = parts.next().unwrap_or_default();
    if rest.is_empty() {
        text.to_string()
    } else {
        format!("{first}, {filler}, {rest}")
    }
}

fn detect_tone(text: &str) -> Option<ToneProfile> {
    let lower = text.to_ascii_lowercase();
    if ["angry", "upset", "complaint", "problem"]
        .iter()
        .any(|w| lower.contains(w))
    {
        Some(ToneProfile::Complaint)
    } else if ["urgent", "immediately", "security", "fraud"]
        .iter()
        .any(|w| lower.contains(w))
    {
        Some(ToneProfile::Escalation)
    } else if ["offer", "price", "deal", "buy"]
        .iter()
        .any(|w| lower.contains(w))
    {
        Some(ToneProfile::Sales)
    } else {
        None
    }
}

struct PunctuationProfile {
    has_sentence_end: bool,
    long_response: bool,
}

fn punctuation_profile(text: &str) -> PunctuationProfile {
    PunctuationProfile {
        has_sentence_end: text.contains('.') || text.contains('?') || text.contains('!'),
        long_response: text.split_whitespace().count() >= 14,
    }
}

fn maybe_schedule_office_fx(engine: &mut AudioHumanizer, speaking: bool, frame_len: usize) {
    if engine.office_fx.is_empty() {
        return;
    }
    engine.samples_until_fx = engine.samples_until_fx.saturating_sub(frame_len);
    if engine.samples_until_fx == 0 {
        if !speaking && engine.active_layers.len() < 2 && engine.rng.chance(20) {
            let index = engine
                .rng
                .range_usize(0, engine.office_fx.len().saturating_sub(1));
            engine.active_layers.push(ActiveLayer {
                samples: engine.office_fx[index].clone(),
                cursor: 0,
                gain: OFFICE_FX_GAIN,
            });
        }

        let min = (SAMPLE_RATE as f32 * 1.5 * engine.tone.pause_density()) as usize;
        let max = (SAMPLE_RATE as f32 * 6.0 * engine.tone.pause_density()) as usize;
        engine.samples_until_fx = engine.rng.range_usize(min.max(800), max.max(1_600));
    }
}

fn is_speaking(frame: &[i16]) -> bool {
    if frame.is_empty() {
        return false;
    }
    let sum = frame
        .iter()
        .map(|sample| (*sample as i32).unsigned_abs() as u64)
        .sum::<u64>();
    sum / frame.len() as u64 > 220
}

fn normalize_voice_frame(frame: &[i16]) -> Vec<i16> {
    let peak = frame
        .iter()
        .map(|sample| sample.unsigned_abs())
        .max()
        .unwrap_or(0);

    let gain = if peak > 28_000 {
        28_000.0 / peak as f32
    } else if peak > 0 && peak < 5_000 {
        (5_000.0 / peak as f32).min(1.4)
    } else {
        1.0
    };

    frame
        .iter()
        .map(|sample| ((*sample as f32) * gain).clamp(i16::MIN as f32, i16::MAX as f32) as i16)
        .collect()
}

fn apply_speed_pitch(frame: &[i16], factor: f32) -> Vec<i16> {
    if frame.is_empty() || (factor - 1.0).abs() < 0.01 {
        return frame.to_vec();
    }

    let target_len = ((frame.len() as f32) / factor).round().max(1.0) as usize;
    (0..target_len)
        .map(|i| {
            let position = i as f32 * factor;
            let left = position.floor() as usize;
            let right = (left + 1).min(frame.len() - 1);
            let frac = position - left as f32;
            let sample = frame[left] as f32 * (1.0 - frac) + frame[right] as f32 * frac;
            sample as i16
        })
        .collect()
}

fn soft_clip_i16(value: f32) -> i16 {
    let normalized = value / i16::MAX as f32;
    let clipped = normalized.tanh() * i16::MAX as f32;
    clipped.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn load_wav_i16(path: impl AsRef<Path>) -> Option<Vec<i16>> {
    let path = path.as_ref();
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) => {
            warn!("failed reading humanization WAV {:?}: {}", path, e);
            return None;
        }
    };
    parse_wav_i16(&bytes).or_else(|| {
        warn!("unsupported humanization WAV {:?}", path);
        None
    })
}

fn parse_wav_i16(bytes: &[u8]) -> Option<Vec<i16>> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }

    let mut offset = 12;
    let mut channels = 0_u16;
    let mut sample_rate = 0_u32;
    let mut bits_per_sample = 0_u16;
    let mut data = None::<&[u8]>;

    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let len = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().ok()?) as usize;
        offset += 8;
        if offset + len > bytes.len() {
            break;
        }

        match id {
            b"fmt " if len >= 16 => {
                let audio_format = u16::from_le_bytes(bytes[offset..offset + 2].try_into().ok()?);
                channels = u16::from_le_bytes(bytes[offset + 2..offset + 4].try_into().ok()?);
                sample_rate = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().ok()?);
                bits_per_sample =
                    u16::from_le_bytes(bytes[offset + 14..offset + 16].try_into().ok()?);
                if audio_format != 1 {
                    return None;
                }
            }
            b"data" => data = Some(&bytes[offset..offset + len]),
            _ => {}
        }
        offset += len + (len % 2);
    }

    let data = data?;
    if bits_per_sample != 16 || channels == 0 {
        return None;
    }

    let samples = data
        .chunks_exact(2 * channels as usize)
        .map(|frame| i16::from_le_bytes([frame[0], frame[1]]))
        .collect::<Vec<_>>();

    if sample_rate == SAMPLE_RATE {
        Some(samples)
    } else {
        Some(resample_linear(&samples, sample_rate, SAMPLE_RATE))
    }
}

fn resample_linear(samples: &[i16], source_rate: u32, target_rate: u32) -> Vec<i16> {
    if samples.is_empty() || source_rate == 0 {
        return Vec::new();
    }
    let target_len = samples.len() * target_rate as usize / source_rate as usize;
    (0..target_len.max(1))
        .map(|i| {
            let position = i as f32 * source_rate as f32 / target_rate as f32;
            let left = position.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let frac = position - left as f32;
            let sample = samples[left] as f32 * (1.0 - frac) + samples[right] as f32 * frac;
            sample as i16
        })
        .collect()
}

fn ms_to_samples(ms: u64) -> usize {
    (ms * SAMPLE_RATE as u64 / 1_000) as usize
}

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .and_then(|value| match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(default)
}

fn env_list(name: &str) -> Vec<String> {
    env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn hash_seed(value: &str) -> u64 {
    value.bytes().fold(0x9e37_79b9_7f4a_7c15, |hash, byte| {
        hash.rotate_left(5) ^ byte as u64
    })
}

struct SmallRng {
    state: u64,
}

impl SmallRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 32) as u32
    }

    fn chance(&mut self, percent: u32) -> bool {
        self.next_u32() % 100 < percent
    }

    fn range_u64(&mut self, min: u64, max: u64) -> u64 {
        if max <= min {
            return min;
        }
        min + (self.next_u32() as u64 % (max - min + 1))
    }

    fn range_usize(&mut self, min: usize, max: usize) -> usize {
        if max <= min {
            return min;
        }
        min + (self.next_u32() as usize % (max - min + 1))
    }
}
