use base64::{engine::general_purpose, Engine as _};
use webrtc_vad::{SampleRate, Vad, VadMode};

pub(crate) const MEDIA_FRAME_MS: u32 = 20;
pub(crate) const TWILIO_ULAW_FRAME_BYTES: usize = 160;
pub(crate) const ULAW_SAMPLE_RATE: u32 = 8_000;
pub(crate) const ULAW_BYTES_PER_MS: u32 = ULAW_SAMPLE_RATE / 1_000;

pub(crate) fn ulaw_base64_duration_ms(payload: &str) -> Option<u64> {
    let bytes = general_purpose::STANDARD.decode(payload).ok()?;
    Some(bytes.len() as u64 / ULAW_BYTES_PER_MS as u64)
}

pub(crate) fn pcm_s16le_to_i16(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect()
}

pub(crate) fn pcm_i16_to_ulaw(samples: &[i16]) -> Vec<u8> {
    samples
        .iter()
        .map(|sample| linear_i16_to_ulaw(*sample))
        .collect()
}

pub(crate) fn ulaw_bytes_to_pcm_i16(bytes: &[u8]) -> Vec<i16> {
    bytes
        .iter()
        .map(|sample| ulaw_to_linear_i16(*sample))
        .collect()
}

pub(crate) fn parse_playback_mark(name: &str) -> Option<(u64, u64)> {
    let mut parts = name.split('-');

    if parts.next()? != "assistant" {
        return None;
    }

    let generation = parts.next()?.parse().ok()?;
    let audio_ms = parts.next()?.parse().ok()?;

    Some((generation, audio_ms))
}

pub(crate) struct WebRtcBargeInVad {
    vad: Vad,
}

impl WebRtcBargeInVad {
    pub(crate) fn new() -> Self {
        Self {
            vad: Vad::new_with_rate_and_mode(SampleRate::Rate8kHz, VadMode::Aggressive),
        }
    }

    pub(crate) fn is_speech_ulaw_bytes(&mut self, bytes: &[u8]) -> bool {
        let samples = bytes
            .iter()
            .map(|&b| ulaw_to_linear_i16(b))
            .collect::<Vec<_>>();

        samples
            .chunks(160)
            .any(|frame| frame.len() == 160 && self.vad.is_voice_segment(frame).unwrap_or(false))
    }
}

fn ulaw_to_linear_i16(u_val: u8) -> i16 {
    const BIAS: i16 = 0x84;

    let u_val = !u_val;
    let sign = u_val & 0x80;
    let exponent = (u_val >> 4) & 0x07;
    let mantissa = u_val & 0x0F;

    let mut sample = ((mantissa as i16) << 3) + BIAS;
    sample <<= exponent as i16;
    sample -= BIAS;

    if sign != 0 {
        -sample
    } else {
        sample
    }
}

fn linear_i16_to_ulaw(sample: i16) -> u8 {
    const BIAS: i32 = 0x84;
    const CLIP: i32 = 32635;

    let mut sample = sample as i32;
    let sign = if sample < 0 {
        sample = -sample;
        0x80
    } else {
        0x00
    };

    if sample > CLIP {
        sample = CLIP;
    }

    sample += BIAS;

    let mut exponent = 7;
    let mut exp_mask = 0x4000;
    while exponent > 0 && (sample & exp_mask) == 0 {
        exponent -= 1;
        exp_mask >>= 1;
    }

    let mantissa = (sample >> (exponent + 3)) & 0x0f;
    !(sign | (exponent << 4) | mantissa) as u8
}
