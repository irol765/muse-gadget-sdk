"""p4 car voice loop (single-shot): wake -> one round -> idle.

p4-specific module for the WK-155 car terminal. It wires three pieces that
already exist: the sherpa wake-word subprocess, the BT600 button daemon
(FIFO events), and the fork's chat path (LinkSession + send_voice).

Finalized flow (2026-11-01): wake -> record -> send the recording as a
voice note -> Muse transcribes it in the cloud -> the text reply is
synthesized by the TTS leg and played on the BT600. After a formal
reply the continuous window opens (1 s pause, beep, 5 s window); a
window trigger starts another round, a window timeout says goodbye
and returns to idle. The reply lands in
the main chat, as a voice note's reply does.

Deliberate v1 choices:
- Continuous window: after a formal reply (reason ok) only, 1 s pause
  then beep then a 5 s window counted from the end of the beep; voice
  (energy gate), wake word, hook press, or pair starts another round,
  window mute is ignored, timeout speaks a goodbye and idles.
- One resident capture (CaptureHub) feeds everything: a single arecord
  writes a 5 s ring, a wake tap, and per-round record queues. The BT600
  capture device is single-open (Pi3 lesson), so rounds never start a
  second arecord; a round starts from the ring's last 1 s (pre-roll)
  instead of from silence.
- Each round uses a FRESH LinkSession (a session runs once) with
  registration on, the same shape the voice-note test proved. Traffic
  goes direct: p4's TUN routes it by Mihomo's existing rules (corrected
  2026-11-01: no forced GLOBAL, no 7890 proxy).
- Recording stops on the gateway VAD (Silero, sidecar process): a
  5-frame window with >= 3 voice frames latches speech; after speech,
  2 s of non-voice ends it (user-set 2026-11-02: 2 s keeps slow
  speakers' tolerance; the hub default was 1 s). 10 s with no speech
  discards it, 30 s is the hard cap (24 kHz mono stays under the
  2 MiB voice-note limit). Without a VAD sidecar, the energy rule
  (RMS >= 100, same frame counts) is the fallback, never the primary.
- TTS leg: direct Azure synthesis first (voice zh-CN-YunxiNeural,
  15 s timeout), Edge read-aloud synthesis second (free, same
  Yunxi voice).
  No local synthesis fallback (WK-230): both routes failing only fails
  the speak step, with both errors logged; the round still ends
  cleanly and the reply text is kept for the diagnostic path.
- Filler (copied from Pi3, not re-synthesized): /root/filler/ (37 WAVs
  + phrases.json) + /root/filler.py (pools + shuffle-deck pick).
  Receipt plays after recording ends; waiting plays every 10 s
  during the reply wait (first at wait+10 s, Pi3 timing), killed
  before the formal reply plays; a close phrase plays on
  wait-mute hang-up. Playback goes through P4Audio's single playback
  device (filler.py's own aplay play() is not used: it would fight
  P4Audio's held device). Wake prompt wired for wake rounds: it
  starts concurrently with registration, before the record window
  opens, so the user hears it at once; speech during the cue is
  kept: recording starts from the ring's
  1 s pre-roll (the cue tail itself is in the pre-roll, as on Pi3). Pair rounds keep the
  "please speak" prompt; chained rounds keep it too. Window prompts are not wired: single-shot
  has no window. Filler is best-effort: a missing filler file or
  module never fails a round.
- Button rules (v1):
  idle: wake word / pair / hook-press -> round (pair rounds play the
  "please speak" prompt first); mute -> ignore.
  record: mute -> discard the recording, back to idle; pair -> submit
  early (too little speech discards instead); hook -> ignore.
  wait: mute -> hang up + close phrase, back to idle; pair -> abandon
  and chain a fresh round with prompt; hook -> ignore.
  play: mute -> stop playback, back to idle; pair -> stop playback and
  chain a fresh round with prompt; hook -> ignore.
- Nothing secret is ever logged: no tokens, no pairing content, no VM
  credentials, no reply text in production logs (only a char count).
  Logs carry counts, reasons, and device names only.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import glob
import importlib.util
import io
import logging
import os
import queue
import random
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import wave
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from musegadget import config, identity, muse_api
from musegadget.link_client import DeviceDescription, LinkSession
from musegadget.service import DEFAULT_NOISE_HOST
from musegadget.voice_audio import WavAudio
from musegadget.voice_call import FRAME_BYTES, SAMPLE_RATE

log = logging.getLogger("musegadget.p4_loop")

# p4 defaults (overridable by CLI flags).
DEFAULT_MODEL = "/root/wake-eval/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
DEFAULT_WAKE_SCRIPT = "/root/wake/sherpa-wake.py"
DEFAULT_VAD_SCRIPT = "/root/wake/vad-sidecar.py"
DEFAULT_VAD_MODEL = "/root/wake/silero_vad.onnx"
DEFAULT_FIFO = "/root/voiceloop/button.fifo"
DEFAULT_PROMPT = "/root/filler/haoqingshuo.wav"
DEFAULT_ARECORD_DEVICE = "hw:BT600,0"
DEFAULT_APLAY_DEVICE = "plughw:BT600,0"

# TTS leg: direct Azure first (paid primary), Edge read-aloud second
# (free backup, same Yunxi voice). No local synthesis: a reply text
# only exists when the network is up, so an offline Piper voice could
# never speak it either (WK-230). Both routes failing means silence
# plus a log line naming both errors.
TTS_KEY_FILE = "/root/voiceloop/.azure-key"
TTS_ENDPOINT = "https://eastasia.tts.speech.microsoft.com/cognitiveservices/v1"
TTS_VOICE = "zh-CN-YunxiNeural"
TTS_TIMEOUT_S = 15
REPLY_TIMEOUT_S = 180
MAX_SUBSCRIBE_SKIPS = 10

# Failure feedback phrases (Pi3 live wording, loop.py:1263/1358):
# a stream/timeout failure says "no reply yet", a discarded/empty
# round says "did not catch that".
NO_REPLY_TEXT = "没等到回复，请稍后再试"
NOT_HEARD_TEXT = "没听清，请再说一遍"

# Filler (copied from Pi3, not re-synthesized): pools + shuffle-deck
# pick come from Pi3's filler module; playback goes through P4Audio's
# single playback device, never filler.py's own aplay play() (it would
# fight P4Audio's held device).
FILLER_PATH = "/root/filler.py"
FILLER_DIR = "/root/filler"
WAIT_FILLER_S = 10.0


def resample_pcm(data: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resample of mono int16 PCM. Length-exact for
    40 ms frames (640 @16k <-> 960 @24k). Pure Python (no numpy: the
    package must not gain a new dependency) and unit-tested."""
    if not data or src_rate == dst_rate:
        return data
    import struct

    n = len(data) // 2
    samples = struct.unpack(f"<{n}h", data[: n * 2])
    out_len = max(1, round(n * dst_rate / src_rate))
    out = []
    for i in range(out_len):
        pos = i * src_rate / dst_rate
        j = int(pos)
        frac = pos - j
        if j + 1 < n:
            value = samples[j] * (1.0 - frac) + samples[j + 1] * frac
        else:
            value = samples[j]
        out.append(max(-32768, min(32767, int(round(value)))))
    return struct.pack(f"<{len(out)}h", *out)


def parse_button_line(line: str) -> str | None:
    """Map one FIFO line from the button daemon to an event name."""
    line = line.strip()
    if line == "BUTTON_PAIR":
        return "pair"
    if line == "BUTTON_MUTE":
        return "mute"
    if line == "PRESS":
        return "hook-press"
    if line == "RELEASE":
        return "hook-release"
    return None


def next_button(button_events: queue.Queue) -> str | None:
    """One queued button event, or None. Callers decide what it means."""
    try:
        return button_events.get_nowait()
    except queue.Empty:
        return None


def frame_rms(frame: bytes) -> float:
    """RMS of one mono int16 PCM frame (pure Python, no new dependency)."""
    n = len(frame) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", frame[: n * 2])
    return (sum(v * v for v in samples) / n) ** 0.5


def wrap_wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    """Wrap mono int16 PCM as a WAV byte string."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        handle.writeframes(pcm)
    return buffer.getvalue()


def read_tts_wav(data: bytes) -> tuple[bytes, int]:
    """Parse synthesized WAV bytes to (pcm, rate). Stereo keeps the left
    channel; anything but 16-bit PCM fails."""
    buffer = io.BytesIO(data)
    with wave.open(buffer, "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        pcm = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError("TTS output is not 16-bit PCM")
    if channels == 2:
        samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
        pcm = struct.pack(f"<{len(samples) // 2}h", *samples[0::2])
    elif channels != 1:
        raise ValueError("TTS output channel count unsupported")
    return pcm, rate


class EnergyStop:
    """Energy-only recording stop, mirroring the Pi3 stop numbers
    without webrtcvad: RMS >= 100 counts as speech, 0.3 s of speech is
    required, 2 s of continuous silence submits, 10 s without speech
    discards, 30 s is the hard cap. Frames are 40 ms. The threshold
    is 100, not Pi3's 1000: p4's BT600 mic level measured 2026-11-02
    peaks at 362 RMS for speech (silence peaks at 29), so 1000 never
    trips on p4."""

    RMS_THRESHOLD = 100.0
    MIN_SPEECH_FRAMES = 8
    SILENCE_STOP_FRAMES = 50
    NO_SPEECH_FRAMES = 250
    MAX_FRAMES = 750

    def __init__(self) -> None:
        self.speech = 0
        self.silence = 0
        self.total = 0

    def feed(self, rms: float) -> str:
        self.total += 1
        if rms >= self.RMS_THRESHOLD:
            self.speech += 1
            self.silence = 0
        elif self.speech:
            self.silence += 1
        if self.speech >= self.MIN_SPEECH_FRAMES and self.silence >= self.SILENCE_STOP_FRAMES:
            return "submit"
        if self.speech == 0 and self.total >= self.NO_SPEECH_FRAMES:
            return "discarded"
        if self.total >= self.MAX_FRAMES:
            return "submit" if self.speech >= self.MIN_SPEECH_FRAMES else "discarded"
        return "continue"


class VadStop:
    """Gateway-VAD recording stop (hub Silero rule, per-frame verdicts
    from the sidecar): a 5-frame window with >= 3 voice frames latches
    speech; after speech, 2 s (50 frames) of non-voice submits
    (user-set 2026-11-02; the hub default was 1 s). 10 s without
    speech discards; 30 s caps. `.speech` counts voice frames, so the
    pair/hook-release early-submit check works unchanged. Frames are
    40 ms. The window spans 5 sidecar frames (200 ms) where the hub
    spans 5 Silero windows (160 ms); the counts, not the span, are
    what the task pins down."""

    WINDOW = 5
    WINDOW_VOICE = 3
    SILENCE_FRAMES = 50  # 2 s, user-set 2026-11-02 (was hub 1 s)
    NO_SPEECH_FRAMES = 250
    MAX_FRAMES = 750

    def __init__(self) -> None:
        self.speech = 0
        self.total = 0
        self._window: collections.deque = collections.deque(maxlen=self.WINDOW)
        self._latched = False
        self._last_voice = -1

    def _have_voice(self) -> bool:
        return sum(self._window) >= self.WINDOW_VOICE

    def feed(self, is_voice: bool) -> str:
        self.total += 1
        if is_voice:
            self.speech += 1
        self._window.append(1 if is_voice else 0)
        if self._have_voice():
            self._latched = True
            self._last_voice = self.total
        if (self._latched and not self._have_voice()
                and self.total - self._last_voice >= self.SILENCE_FRAMES):
            return "submit"
        if not self._latched and self.total >= self.NO_SPEECH_FRAMES:
            return "discarded"
        if self.total >= self.MAX_FRAMES:
            return "submit" if self._latched else "discarded"
        return "continue"


def build_ssml(text: str) -> bytes:
    """SSML for the direct Azure call (voice zh-CN-YunxiNeural)."""
    body = (
        "<speak version='1.0' xml:lang='zh-CN'>"
        f"<voice name='{TTS_VOICE}'>{xml_escape(text)}</voice></speak>"
    )
    return body.encode("utf-8")


def yunxi_synthesize(text: str) -> bytes:
    """Direct Azure synthesis (no SDK, stdlib urllib only, same shape as
    the Pi3 TTS server). Raises on a missing key file or any failure;
    the caller falls back to the Edge route."""
    key = Path(TTS_KEY_FILE).read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError("TTS key file is empty")
    request = urllib.request.Request(
        TTS_ENDPOINT,
        data=build_ssml(text),
        headers={
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "riff-22050hz-16bit-mono-pcm",
            "User-Agent": "musegadget-p4-loop",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=TTS_TIMEOUT_S) as response:
        data = response.read()
    if not data.startswith(b"RIFF"):
        raise ValueError("TTS endpoint did not return WAV audio")
    return data


def edge_synthesize(text: str) -> bytes:
    """Free backup synthesis via Edge read-aloud (same Yunxi voice
    as the primary route). Raises on any failure; there is no
    third leg."""
    from musegadget.edge_tts import synthesize_edge_wav

    return synthesize_edge_wav(text)


def _synthesize_with_fallback(text: str) -> bytes:
    """Primary Azure, backup Edge. Both failing raises a RuntimeError
    carrying both routes' errors so the failure is never silent."""
    try:
        return yunxi_synthesize(text)
    except Exception as primary_exc:
        primary_error = primary_exc  # `as` names are deleted on block exit
        log.warning("direct TTS failed (%r), falling back to Edge TTS",
                    primary_error)
    try:
        return edge_synthesize(text)
    except Exception as backup_exc:
        log.error("Edge TTS also failed (%r); reply will not be spoken",
                  backup_exc)
        raise RuntimeError(
            f"all TTS routes failed: primary Azure: {primary_error!r}; "
            f"backup Edge: {backup_exc!r}") from backup_exc


class ReplyCollector:
    """Accumulate one text reply from subscribe_chat envelopes (same
    parsing the voice-note test proved): delta.text_append accumulates,
    delta.message_done / message.assistant completes, user-role events
    are skipped.

    WK-226: a completion carrying no text at all (an agent turn that
    ended without a visible reply) does NOT complete the round. It is
    counted in `empty_swallowed`, logged once per message, and the
    collector keeps waiting for a later message with actual text;
    only the caller's reply timeout ends a round that never gets one.
    """

    def __init__(self) -> None:
        self.text = ""
        self.message_id: str | None = None
        self.done = False
        self.empty_swallowed = 0
        self._swallowed_ids: set = set()

    def feed(self, event) -> None:
        if not isinstance(event, dict) or self.done:
            return
        name = event.get("event") or event.get("name") or event.get("type")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else event
        if payload.get("role") == "user":
            return
        mid = payload.get("message_id") or payload.get("id")
        if name == "delta.text_append" and isinstance(payload.get("text"), str):
            self.text += payload["text"]
            if mid and self.message_id is None:
                self.message_id = mid
        elif name in ("delta.message_done", "message.assistant"):
            text = payload.get("text")
            if not (isinstance(text, str) and text):
                text = payload.get("display_text")
            if isinstance(text, str) and text and not self.text:
                self.text = text
            if self.text:
                if mid and self.message_id is None:
                    self.message_id = mid
                self.done = True
            elif mid is None or mid not in self._swallowed_ids:
                # Empty completion: an agent turn ended with nothing
                # to say. Not this round's answer - swallow it, keep
                # waiting, and leave a countable trace in the log.
                if mid is not None:
                    self._swallowed_ids.add(mid)
                self.empty_swallowed += 1
                log.info("reply collector: empty completion swallowed "
                         "(%d, msg %s); still waiting",
                         self.empty_swallowed, mid)
                self.message_id = None


def load_filler_module(path: str = FILLER_PATH):
    """Load Pi3's copied filler module (pools + shuffle-deck pick) or
    None when it is absent/unreadable. load_pools/pick/load_fixed are used;
    its aplay play() is deliberately not used (see FILLER_PATH note)."""
    try:
        spec = importlib.util.spec_from_file_location("pi_filler", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        log.warning("filler module unavailable (%s)", type(exc).__name__)
        return None


def wav_bytes_to_frames(data: bytes) -> list[bytes]:
    """Decode a synthesized/filler WAV to 24 kHz mono frames."""
    pcm, rate = read_tts_wav(data)
    src_frame = max(1, int(rate * 0.04)) * 2  # bytes per 40 ms at src rate
    return [
        resample_pcm(pcm[offset:offset + src_frame], rate, SAMPLE_RATE)
        for offset in range(0, len(pcm), src_frame)
    ]


async def play_wav_frames(frames: list[bytes], audio, button_events: queue.Queue,
                        wake_queue: queue.Queue | None = None,
                        hook_interrupt: bool = False) -> str:
    """Feed 24 kHz frames through `audio`'s frame sink with 40 ms pacing
    and button polling. Returns "ok", "cancelled-mute", "cancelled-pair",
    "interrupted", or "audio_failed". Already-queued frames drain briefly
    after an interrupt (aplay's own buffer). Mute and pair always
    interrupt. A wake word interrupts only when `wake_queue` is given
    (formal reply playback); a hook press interrupts only when
    `hook_interrupt` is set (formal reply playback). Without either,
    behavior is exactly as before (receipt, close phrase, wake cue:
    hook edges ignored, wake words only queued). Any interrupt logs
    how many milliseconds were sent (frames handed to the sink; the
    sink's own buffer still drains briefly after, as before)."""
    written = 0
    for frame in frames:
        while True:
            event = next_button(button_events)
            if event is None:
                break
            if event == "mute":
                audio.flush()
                log.info("playback interrupted after %d ms sent", written * 40)
                return "cancelled-mute"
            if event == "pair":
                audio.flush()
                log.info("playback interrupted after %d ms sent", written * 40)
                return "cancelled-pair"
            if event == "hook-press" and hook_interrupt:
                audio.flush()
                log.info("playback interrupted after %d ms sent", written * 40)
                return "interrupted"
            # Other hook edges do not interrupt playback.
        if wake_queue is not None:
            try:
                wake_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                audio.flush()
                log.info("playback interrupted after %d ms sent", written * 40)
                return "interrupted"
        for _ in range(40):  # ~2 s of backpressure retries per frame
            try:
                audio.write(frame)
                break
            except OSError:
                await asyncio.sleep(0.05)
        else:
            log.warning("playback failed: sink stayed full")
            return "audio_failed"
        written += 1
        await asyncio.sleep(0.04)
    return "ok"


async def play_pool_cue(filler, pools, pool_name, audio, button_events: queue.Queue) -> str | None:
    """Play one shuffle-picked cue from a filler pool, or None when the
    pool is missing, the pick is empty, or the WAV cannot be decoded.
    Returns play_wav_frames' reason otherwise ("ok", "cancelled-mute",
    "cancelled-pair", "audio_failed"). Best-effort: a missing cue never
    fails the round; the caller decides what a cancellation means."""
    if filler is None or pools is None:
        return None
    pool = pools.get(pool_name)
    if not pool:
        return None
    path = filler.pick(pool)
    if not path:
        return None
    try:
        frames = wav_bytes_to_frames(Path(path).read_bytes())
    except Exception as exc:
        log.warning("%s cue skipped (%s)", pool_name, type(exc).__name__)
        return None
    if not frames:
        return None
    return await play_wav_frames(frames, audio, button_events)


async def speak_text(text: str, audio, button_events: queue.Queue,
                   wake_queue: queue.Queue | None = None,
                   hook_interrupt: bool = False) -> str:
    """Synthesize (Azure, then Edge) and play through `audio`'s 24 kHz
    frame sink. Returns "ok", "cancelled-mute", "cancelled-pair",
    "interrupted", "tts_failed", or "audio_failed". `wake_queue` and
    `hook_interrupt` enable formal-playback interrupts (step 3); callers
    that leave them unset get the old mute/pair-only behavior."""
    try:
        wav = await asyncio.to_thread(_synthesize_with_fallback, text)
    except Exception as exc:
        log.warning("speak failed at synthesis (%s: %s)",
                    type(exc).__name__, exc)
        return "tts_failed"
    try:
        frames = wav_bytes_to_frames(wav)
    except Exception as exc:
        log.warning("speak failed at WAV parse (%s)", type(exc).__name__)
        return "tts_failed"
    return await play_wav_frames(frames, audio, button_events,
                                 wake_queue=wake_queue,
                                 hook_interrupt=hook_interrupt)


# Fixed failure phrases and their pre-recorded clips in the filler
# pool (phrases.json "pool": "fixed"). The clips carry the main-route
# voice, generated once; a fixed phrase is never live-synthesized
# while its clip exists (live synthesis stays for reply text, and as
# the fallback below when a clip is missing - a failure must never
# go silent either).
FIXED_FEEDBACK_CLIPS = {
    NOT_HEARD_TEXT: "fixed_not_heard",
    NO_REPLY_TEXT: "fixed_no_reply",
}


async def play_fixed_cue(name: str, audio, button_events: queue.Queue) -> str | None:
    """Play one named fixed-phrase clip from the filler pool, or None
    when the pool or the clip is unavailable. Returns play_wav_frames'
    reason otherwise. Mirrors play_pool_cue, addressed by name."""
    filler = load_filler_module()
    if filler is None:
        return None
    try:
        path = filler.load_fixed().get(name)
    except Exception as exc:
        log.warning("fixed cue %s skipped (%s)", name, type(exc).__name__)
        return None
    if not path:
        return None
    try:
        frames = wav_bytes_to_frames(Path(path).read_bytes())
    except Exception as exc:
        log.warning("fixed cue %s skipped (%s)", name, type(exc).__name__)
        return None
    if not frames:
        return None
    log.info("fixed cue %s (%s)", name, Path(path).name)
    return await play_wav_frames(frames, audio, button_events)


async def play_failure_feedback(text: str, audio, button_events: queue.Queue) -> str | None:
    """Play one failure phrase: its pre-recorded clip when the phrase
    is a fixed one, live synthesis otherwise (and as the fallback when
    the clip is missing).

    Returns "cancelled-pair" when the user chains a fresh round during
    the phrase, else None: the caller keeps its original reason.
    A failed synth or playback only logs inside speak_text; this
    never raises and never masks the original reason.
    """
    clip = FIXED_FEEDBACK_CLIPS.get(text)
    if clip:
        try:
            reason = await play_fixed_cue(clip, audio, button_events)
        except Exception as exc:
            log.warning("fixed feedback failed (%s)", type(exc).__name__)
            reason = None
        if reason is not None:
            if reason == "cancelled-pair":
                return "cancelled-pair"
            return None
        log.warning("fixed clip %s unavailable, synthesizing instead", clip)
    try:
        reason = await speak_text(text, audio, button_events)
    except Exception as exc:
        log.warning("failure feedback failed (%s)", type(exc).__name__)
        return None
    if reason == "cancelled-pair":
        return "cancelled-pair"
    return None


def _load_speexdsp():
    """ctypes handle for libspeexdsp (system package libspeexdsp1);
    raises OSError when it is not installed."""
    import ctypes.util

    for name in (ctypes.util.find_library("speexdsp"), "libspeexdsp.so.1"):
        if name:
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
    raise OSError("libspeexdsp not found")


class LinearAec:
    """Speex MDF linear echo canceller (thinksmart speex_aec.py shape):
    16 kHz mono int16, 20 ms blocks, 500 ms tail. Only the linear
    estimate is subtracted, so a voice talking over the reply stays:
    WebRTC APM's suppressor would eat it, which is why the wake word
    listens to this during playback, not to APM output. Raises
    OSError when libspeexdsp is missing; callers treat that as
    'no AEC, wake tap stays raw'. Only stdlib (ctypes)."""

    FRAME = 320  # samples per block
    TAIL_MS = 500
    _SAMPLING_RATE_CTL = 24  # SPEEX_ECHO_SET_SAMPLING_RATE

    def __init__(self, rate: int = 16000) -> None:
        import ctypes

        lib = _load_speexdsp()
        lib.speex_echo_state_init.restype = ctypes.c_void_p
        lib.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
        lib.speex_echo_cancellation.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.speex_echo_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.speex_echo_state_reset.argtypes = [ctypes.c_void_p]
        self._lib = lib
        self._st = lib.speex_echo_state_init(self.FRAME, int(rate * self.TAIL_MS / 1000))
        if not self._st:
            raise OSError("speex echo state init failed")
        rate_arg = ctypes.c_int(rate)
        lib.speex_echo_ctl(self._st, self._SAMPLING_RATE_CTL, ctypes.byref(rate_arg))
        self._out = (ctypes.c_short * self.FRAME)()

    def process(self, near: bytes, far: bytes) -> bytes:
        """One 20 ms block: near mic, far reference, both int16 mono
        of exactly FRAME samples. Returns the cancelled block."""
        import ctypes

        near_arr = (ctypes.c_short * self.FRAME).from_buffer_copy(near)
        far_arr = (ctypes.c_short * self.FRAME).from_buffer_copy(far)
        self._lib.speex_echo_cancellation(self._st, near_arr, far_arr, self._out)
        return bytes(self._out)

    def reset(self) -> None:
        self._lib.speex_echo_state_reset(self._st)

    def close(self) -> None:
        st = getattr(self, "_st", None)
        self._st = None
        if st:
            self._lib.speex_echo_state_destroy(st)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class CaptureHub:
    """One resident capture feeding a ring buffer, a wake tap, and
    per-round record queues.

    The BT600 capture device is single-open, so exactly one arecord
    runs for the whole process. Its reader thread appends every 40 ms
    chunk to a 5 s ring, copies it to the wake tap while armed, and
    fans it out to every subscribed record queue (a full queue drops
    its oldest chunk). A round starts from the ring's last 1 s
    (pre-roll) instead of from silence: speech during the wake prompt
    and registration is kept, not drained. If arecord dies, the reader
    relaunches it within 1 s (0.2 s delay); a round that then sees no
    chunk for 2 s treats capture as broken instead of hanging.
    Only stdlib; no new dependencies.
    """

    CAPTURE_RATE = 16000
    CAPTURE_CHUNK = int(CAPTURE_RATE * 0.04) * 2  # 40 ms mono int16 = 1280 B
    RING_CHUNKS = 125  # 5 s
    PRE_ROLL_CHUNKS = 25  # 1 s
    RECORD_QUEUE_CHUNKS = 125  # 5 s of slack per round
    WAKE_QUEUE_CHUNKS = 25  # 1 s of slack for the wake writer
    RELAUNCH_DELAY_S = 0.2
    STALL_S = 2.0  # no chunk this long means capture is broken

    def __init__(self, device: str = DEFAULT_ARECORD_DEVICE, spawn=None) -> None:
        self._device = device
        self._spawn = spawn or self._default_spawn
        self._lock = threading.Lock()
        self._ring: collections.deque = collections.deque(maxlen=self.RING_CHUNKS)
        self._record_queues: set = set()
        self._wake_queue: queue.Queue = queue.Queue(maxsize=self.WAKE_QUEUE_CHUNKS)
        self._wake_armed = False
        self._aec = None  # LinearAec, built lazily on first enable
        self._aec_active = False
        self._far = bytearray()  # 16 kHz mono int16 playback reference
        self._stopped = threading.Event()
        self._started = False
        self._reader: threading.Thread | None = None
        self._proc = None
        self._last_chunk_t: float | None = None

    def _default_spawn(self):
        return subprocess.Popen(
            ["arecord", "-D", self._device, "-f", "S16_LE",
             "-r", str(self.CAPTURE_RATE), "-c", "1", "-t", "raw"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._stopped.clear()
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    @property
    def last_chunk_t(self) -> float | None:
        return self._last_chunk_t

    @property
    def wake_armed(self) -> bool:
        with self._lock:
            return self._wake_armed

    def _read_exact(self, stream, size: int) -> bytes | None:
        buf = b""
        while len(buf) < size:
            data = stream.read(size - len(buf))
            if not data:
                return None
            buf += data
        return buf

    def _put_drop_oldest(self, q: queue.Queue, item: bytes) -> None:
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    def _distribute(self, chunk: bytes) -> None:
        with self._lock:
            self._ring.append(chunk)
            if self._wake_armed:
                # Playback only: the wake tap hears the linear-cancelled
                # mic; the ring and record queues always stay raw.
                tap = self._aec_process(chunk) if self._aec_active else chunk
                self._put_drop_oldest(self._wake_queue, tap)
            for q in list(self._record_queues):
                self._put_drop_oldest(q, chunk)
            self._last_chunk_t = time.monotonic()

    @property
    def aec_active(self) -> bool:
        return self._aec_active and self._aec is not None

    def enable_aec(self) -> bool:
        """Arm linear cancellation for the wake tap (formal playback):
        resets the canceller and clears the reference. Missing
        libspeexdsp logs a warning and leaves the tap raw; never raises."""
        with self._lock:
            if self._aec is None:
                try:
                    self._aec = LinearAec()
                except OSError as exc:
                    log.warning("AEC unavailable (%s); wake tap stays raw",
                                type(exc).__name__)
                    return False
            self._aec.reset()
            self._far.clear()
            self._aec_active = True
            return True

    def disable_aec(self) -> None:
        """Stop cancellation (playback over): the tap goes raw again."""
        with self._lock:
            self._aec_active = False
            self._far.clear()

    def feed_far_16k(self, pcm: bytes) -> None:
        """Queue playback reference frames (16 kHz mono int16), exactly
        what the speaker was sent, for the linear canceller."""
        with self._lock:
            if self._aec_active:
                self._far += pcm

    def _aec_process(self, chunk: bytes) -> bytes:
        """Cancel one 40 ms mic chunk against the far reference.
        Caller holds the lock. Far running more than 0.2 s ahead of
        the mic (thinksmart guard: the output callback runs ahead in
        bursts) drops its oldest part; a short reference zero-pads."""
        limit = 6400 + len(chunk)  # 0.2 s at 16 kHz, plus one chunk
        if len(self._far) > limit:
            del self._far[:len(self._far) - limit]
        ref = bytes(self._far[:len(chunk)])
        del self._far[:len(ref)]
        if len(ref) < len(chunk):
            ref += b"\x00" * (len(chunk) - len(ref))
        assert self._aec is not None
        out = bytearray()
        step = LinearAec.FRAME * 2
        for off in range(0, len(chunk), step):
            out += self._aec.process(chunk[off:off + step], ref[off:off + step])
        return bytes(out)

    def _kill_proc(self, proc) -> None:
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=1)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _reader_loop(self) -> None:
        while not self._stopped.is_set():
            try:
                proc = self._spawn()
            except Exception as exc:
                log.warning("capture spawn failed (%s)", type(exc).__name__)
                self._stopped.wait(self.RELAUNCH_DELAY_S)
                continue
            self._proc = proc
            try:
                assert proc.stdout is not None
                while not self._stopped.is_set():
                    chunk = self._read_exact(proc.stdout, self.CAPTURE_CHUNK)
                    if chunk is None:
                        break
                    self._distribute(chunk)
            except Exception as exc:
                log.warning("capture read failed (%s)", type(exc).__name__)
            finally:
                self._kill_proc(proc)
                self._proc = None
            if not self._stopped.is_set():
                self._stopped.wait(self.RELAUNCH_DELAY_S)

    def snapshot(self, count: int = PRE_ROLL_CHUNKS) -> list[bytes]:
        with self._lock:
            ring = list(self._ring)
        return ring[-count:] if count else []

    def subscribe_record(self) -> tuple[queue.Queue, list[bytes]]:
        # Snapshot and registration share one lock: no chunk can slip
        # between the pre-roll and the live feed.
        with self._lock:
            snap = list(self._ring)[-self.PRE_ROLL_CHUNKS:]
            record_queue: queue.Queue = queue.Queue(maxsize=self.RECORD_QUEUE_CHUNKS)
            self._record_queues.add(record_queue)
        return record_queue, snap

    def unsubscribe_record(self, record_queue: queue.Queue) -> None:
        with self._lock:
            self._record_queues.discard(record_queue)

    def arm_wake(self) -> queue.Queue:
        # Clearing first drops any stale feed queued before this arm.
        with self._lock:
            while True:
                try:
                    self._wake_queue.get_nowait()
                except queue.Empty:
                    break
            self._wake_armed = True
        return self._wake_queue

    def disarm_wake(self) -> None:
        with self._lock:
            self._wake_armed = False

    def stop(self) -> None:
        self._stopped.set()
        proc = self._proc
        if proc is not None:
            self._kill_proc(proc)
        reader = self._reader
        if reader is not None:
            reader.join(timeout=2)
        self._reader = None


class P4Audio:
    """Live audio adapter for the voice-note round, mirroring LiveAudio's
    interface (frames/write/flush/close).

    PortAudio cannot open this BT600's output at all (diagnosed 2026-11-01:
    every output open fails "Invalid number of channels", while capture
    only works at 16 kHz mono), so the subprocess path is the only one that
    works on p4. Capture comes from the shared CaptureHub (one resident
    arecord, 1 s pre-roll, no per-round capture process); playback is a
    per-round aplay fed 24 kHz mono through plughw (which converts to
    the device rate). Playback runs in a thread so the event loop never
    blocks."""

    CAPTURE_RATE = CaptureHub.CAPTURE_RATE
    CAPTURE_CHUNK = CaptureHub.CAPTURE_CHUNK

    def __init__(self, record_device: str = DEFAULT_ARECORD_DEVICE,
                 play_device: str = DEFAULT_APLAY_DEVICE,
                 hub: CaptureHub | None = None) -> None:
        # When a hub is given, its device is the capture device and
        # record_device is only kept for call compatibility; when none
        # is given, this round owns a hub built on record_device.
        self._record_device = record_device
        self._play_device = play_device
        self._playback: queue.Queue[bytes] = queue.Queue(maxsize=8)
        self._closed = False
        self._playback_dead = False
        self._record_queue: queue.Queue | None = None
        if hub is None:
            self._hub = CaptureHub(record_device)
            self._hub_owned = True
            self._hub.start()
        else:
            self._hub = hub
            self._hub_owned = False
        try:
            self._aplay = subprocess.Popen(
                ["aplay", "-D", play_device, "-f", "S16_LE",
                 "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw"],
                stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except FileNotFoundError as exc:
            self._close_hub_if_owned()
            raise RuntimeError("aplay not installed") from exc
        time.sleep(0.2)
        if self._aplay.poll() is not None:
            self.close()
            raise RuntimeError(f"aplay exited immediately (rc={self._aplay.returncode})")
        log.info("audio: aplay pid=%d (capture via shared hub)", self._aplay.pid)
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()

    def _playback_loop(self) -> None:
        assert self._aplay.stdin is not None
        try:
            while True:
                pcm = self._playback.get()
                try:
                    self._aplay.stdin.write(pcm)
                    self._aplay.stdin.flush()
                except (BrokenPipeError, OSError):
                    self._playback_dead = True
                    return
        except Exception:
            self._playback_dead = True

    async def frames(self):
        # Pre-roll first: the hub ring's last 1 s, then live chunks.
        # No drain: speech during the prompt and registration is kept.
        record_queue, pre_roll = self._hub.subscribe_record()
        self._record_queue = record_queue
        try:
            for chunk in pre_roll:
                yield resample_pcm(chunk, self.CAPTURE_RATE, SAMPLE_RATE)
            started = time.monotonic()
            while True:
                if self._closed:
                    return
                if self._hub.stopped:
                    try:
                        data = record_queue.get_nowait()
                    except queue.Empty:
                        raise OSError("audio device stopped")
                    yield resample_pcm(data, self.CAPTURE_RATE, SAMPLE_RATE)
                    continue
                try:
                    data = record_queue.get_nowait()
                except queue.Empty:
                    last = self._hub.last_chunk_t
                    now = time.monotonic()
                    if last is not None and now - last > CaptureHub.STALL_S:
                        raise OSError("audio device stopped")
                    if last is None and now - started > CaptureHub.STALL_S:
                        raise OSError("audio device stopped")
                    await asyncio.sleep(0.01)
                    continue
                yield resample_pcm(data, self.CAPTURE_RATE, SAMPLE_RATE)
        finally:
            self._unsubscribe_record()

    def write(self, pcm: bytes) -> None:
        if not pcm or len(pcm) > FRAME_BYTES or len(pcm) % 2:
            raise ValueError("playback needs 1-960 PCM16 samples")
        if self._closed or self._playback_dead:
            raise OSError("audio device stopped")
        try:
            self._playback.put_nowait(pcm)
        except queue.Full:
            raise OSError("audio device playback queue full") from None
        # Far reference for the wake tap's linear canceller, only while
        # formal playback has AEC armed (P4Audio.write serves every
        # playback, but only formal playback arms it). Fed after the
        # successful queue put, so a retried frame is never fed twice;
        # a feed failure must never break playback.
        hub = getattr(self, "_hub", None)
        if hub is not None and getattr(hub, "aec_active", False):
            try:
                hub.feed_far_16k(resample_pcm(pcm, SAMPLE_RATE, 16000))
            except Exception:
                pass

    def flush(self) -> None:
        while True:
            try:
                self._playback.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._unsubscribe_record()
        proc = getattr(self, "_aplay", None)
        if proc is not None:
            # End the stream the way a file playback ends: closing
            # stdin gives aplay EOF, so it drains what is buffered
            # and exits on its own. Signals are only a fallback, and
            # every path waits, so no zombie is left behind and the
            # audio device is never left held by a killed stream.
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=2)
                    except Exception:
                        pass
            except Exception:
                pass
        self._close_hub_if_owned()

    def _close_hub_if_owned(self) -> None:
        if getattr(self, "_hub_owned", False):
            try:
                self._hub.stop()
            except Exception:
                pass

    def _unsubscribe_record(self) -> None:
        record_queue = self._record_queue
        self._record_queue = None
        hub = getattr(self, "_hub", None)
        if record_queue is not None and hub is not None:
            try:
                hub.unsubscribe_record(record_queue)
            except Exception:
                pass


def probe_audio(record_device: str = DEFAULT_ARECORD_DEVICE,
                play_device: str = DEFAULT_APLAY_DEVICE) -> int:
    """Standalone probe: BT600 presence via arecord/aplay plus a 1 s
    capture byte-count check (16 kHz mono raw = 32000 bytes)."""
    ok = True
    for cmd in (["arecord", "-l"], ["aplay", "-l"]):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            lines = [ln for ln in result.stdout.splitlines() if "BT600" in ln]
            print(f"{cmd[0]}: " + ("; ".join(lines) if lines else "BT600 not listed"))
            ok = ok and bool(lines)
        except Exception as exc:
            print(f"{cmd[0]}: failed ({type(exc).__name__})")
            ok = False
    try:
        result = subprocess.run(
            ["arecord", "-D", record_device, "-f", "S16_LE", "-r", "16000",
             "-c", "1", "-t", "raw", "-d", "1"],
            capture_output=True, timeout=10)
        count = len(result.stdout)
        print(f"probe: 1 s capture = {count} bytes (expect 32000)")
        ok = ok and count >= 30000
    except Exception as exc:
        print(f"probe: capture failed ({type(exc).__name__})")
        ok = False
    print(f"probe: playback via {play_device} (plughw converts to device rate)")
    return 0 if ok else 1


class ButtonReader(threading.Thread):
    """Read button-daemon FIFO lines into a queue. Runs as a daemon thread;
    the open() blocks until the daemon opens the write end, which is fine:
    the wake phase does not need buttons to start listening."""

    def __init__(self, fifo: str, events: queue.Queue) -> None:
        super().__init__(daemon=True)
        self._fifo = fifo
        self._events = events

    def run(self) -> None:
        while True:
            try:
                with open(self._fifo, "r") as handle:
                    for line in handle:
                        event = parse_button_line(line)
                        if event is not None:
                            self._events.put(event)
            except FileNotFoundError:
                log.warning("button FIFO missing, retrying")
            except Exception:
                log.warning("button FIFO read failed, retrying")
            time.sleep(1)


class WakePhase:
    """Listen for the wake word through the shared capture hub.

    The sherpa process no longer owns arecord: its stdin is fed by a
    writer thread draining the hub's wake tap. The process lives
    across rounds (started once, stopped at loop exit); only the tap
    is gated at round boundaries: arm() feeds the tap, disarm()
    starves it while the writer thread keeps polling (0.1 s, harmless
    when the tap is empty; re-arm clears stale feed). Arming policy
    lives in run_loop/run_round: idle, playback and reply-wait armed;
    recording disarmed. Detections reach the wake queue whenever the
    tap is armed; acting on in-round detections belongs to the
    interrupt step, not to this phase.
    The sherpa script flushes each detection line, so plain line
    reads see detections promptly.
    """

    def __init__(self, model: str, script: str, device: str, wake_queue: queue.Queue,
                 hub=None) -> None:
        self._model = model
        self._script = script
        # device is retained for call compatibility; capture now comes
        # from the hub, so it is not used here.
        self._device = device
        self._wake_queue = wake_queue
        self._hub = hub
        self._sherpa: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._writer: threading.Thread | None = None
        self._writer_stop = threading.Event()

    def start(self) -> None:
        if self._hub is None:
            raise RuntimeError("WakePhase requires a capture hub")
        if self.alive():
            return
        # A previous writer may still poll an empty tap after a
        # spontaneous death; stop it so only the new writer feeds.
        self._writer_stop.set()
        old_writer = self._writer
        self._writer = None
        if old_writer is not None:
            old_writer.join(timeout=1)
        self._terminate_sherpa()
        self._sherpa = subprocess.Popen(
            [self._script, "-model", self._model, "-cutoff", "0.2",
             "-window", "6", "-step", "10"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert self._sherpa.stdout is not None and self._sherpa.stdin is not None
        self._reader = threading.Thread(
            target=self._read_loop, args=(self._sherpa.stdout,), daemon=True)
        self._reader.start()
        try:
            wake_feed = self._hub.arm_wake()
            self._writer_stop.clear()
            self._writer = threading.Thread(
                target=self._feed_loop, args=(wake_feed, self._sherpa.stdin),
                daemon=True)
            self._writer.start()
        except Exception:
            self._terminate_sherpa()
            raise

    def arm(self) -> None:
        """Feed the wake tap (idle, playback, reply-wait). The writer
        thread keeps running through disarm, so arming only flips the
        hub tap (which clears stale feed first)."""
        if self._hub is not None:
            self._hub.arm_wake()

    def disarm(self) -> None:
        """Starve the wake tap (recording). The writer thread survives
        on empty polls; the sherpa process is untouched."""
        if self._hub is not None:
            self._hub.disarm_wake()

    def _feed_loop(self, feed: queue.Queue, stdin) -> None:
        while not self._writer_stop.is_set():
            try:
                chunk = feed.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                stdin.write(chunk)
                stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                return

    def _read_loop(self, stdout) -> None:
        try:
            for raw in stdout:
                try:
                    line = raw.decode("utf-8", errors="ignore")
                except Exception:
                    continue
                if "WAKE WORD DETECTED" in line:
                    self._wake_queue.put("wake")
        except Exception:
            pass

    def alive(self) -> bool:
        return self._sherpa is not None and self._sherpa.poll() is None

    def _terminate_sherpa(self) -> None:
        proc = self._sherpa
        self._sherpa = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
        if proc is not None:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    def stop(self) -> None:
        self._writer_stop.set()
        writer = self._writer
        self._writer = None
        if writer is not None:
            writer.join(timeout=1)
        if self._hub is not None:
            self._hub.disarm_wake()
        self._terminate_sherpa()


class VadPhase:
    """Resident Silero VAD sidecar (sherpa-venv): stdin 16 kHz PCM
    frames, stdout one 1/0 line per 40 ms frame. Mirrors WakePhase's
    spawn/feed/read shape, but the feed is driven by the record loop
    (each recorded frame is forwarded), not by the hub tap: the tap is
    disarmed during recording, and the record loop already holds every
    frame. Decisions drop oldest when full (a stale verdict is worse
    than a dropped one); a dead sidecar fails the record as
    audio_failed, it never silently falls back mid-round."""

    def __init__(self, script: str, model: str) -> None:
        self._script = script
        self._model = model
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._writer: threading.Thread | None = None
        self._writer_stop = threading.Event()
        self._feed: queue.Queue = queue.Queue(maxsize=25)
        self._decisions: queue.Queue = queue.Queue(maxsize=250)

    def start(self) -> None:
        if self.alive():
            return
        self._writer_stop.set()
        old_writer = self._writer
        self._writer = None
        if old_writer is not None:
            old_writer.join(timeout=1)
        self._terminate()
        self._proc = subprocess.Popen(
            [self._script, "--model", self._model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert self._proc.stdout is not None and self._proc.stdin is not None
        self._drain(self._feed)
        self._drain(self._decisions)
        self._reader = threading.Thread(
            target=self._read_loop, args=(self._proc.stdout,), daemon=True)
        self._reader.start()
        self._writer_stop.clear()
        self._writer = threading.Thread(
            target=self._feed_loop, args=(self._feed, self._proc.stdin),
            daemon=True)
        self._writer.start()

    @staticmethod
    def _drain(q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _terminate(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
        if proc is not None:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    def stop(self) -> None:
        self._writer_stop.set()
        writer = self._writer
        self._writer = None
        if writer is not None:
            writer.join(timeout=1)
        self._terminate()

    def feed_input(self, pcm16: bytes) -> None:
        """Forward one recorded 16 kHz frame; drops oldest when full."""
        try:
            self._feed.put_nowait(pcm16)
        except queue.Full:
            try:
                self._feed.get_nowait()
            except queue.Empty:
                pass
            try:
                self._feed.put_nowait(pcm16)
            except queue.Full:
                pass

    def next_decision(self) -> bool | None:
        """One sidecar verdict, or None when none is queued yet."""
        try:
            return self._decisions.get_nowait() == "1"
        except queue.Empty:
            return None

    def _feed_loop(self, feed: queue.Queue, stdin) -> None:
        while not self._writer_stop.is_set():
            try:
                chunk = feed.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                stdin.write(chunk)
                stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                return

    def _read_loop(self, stdout) -> None:
        try:
            for raw in stdout:
                line = raw.decode("utf-8", errors="ignore").strip()
                if line in ("0", "1"):
                    try:
                        self._decisions.put_nowait(line)
                    except queue.Full:
                        try:
                            self._decisions.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self._decisions.put_nowait(line)
                        except queue.Full:
                            pass
        except Exception:
            pass


def play_prompt(path: str, aplay_device: str) -> None:
    """Play the pair prompt (best effort; a missing file only skips it)."""
    if not Path(path).is_file():
        log.warning("prompt file missing, skipping")
        return
    try:
        result = subprocess.run(
            ["aplay", "-D", aplay_device, path],
            timeout=15,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("prompt played rc=%d", result.returncode)
    except Exception:
        log.warning("prompt playback failed")


def play_pair_prompt(filler, pools, prompt_path: str, aplay_device: str) -> None:
    """Pair-round prompt: one wake-pool phrase via aplay, else the fixed prompt.

    `filler`/`pools` come from run_loop's one-time load, so the filler
    module's shuffle deck survives across prompts. A missing module,
    pool, or file falls back to the fixed prompt; the round never
    fails here. Chained rounds do not use this: they keep the fixed
    prompt.
    """
    if filler is not None and pools is not None:
        wake_files = pools.get("wake") or []
        if wake_files:
            try:
                path = filler.pick(wake_files)
            except Exception:
                path = None
            if path:
                play_prompt(path, aplay_device)
                return
    play_prompt(prompt_path, aplay_device)


def drain(events: queue.Queue) -> None:
    """Drop stale button events queued before a round starts."""
    while True:
        try:
            events.get_nowait()
        except queue.Empty:
            return


WINDOW_S = 5.0
WINDOW_PAUSE_S = 1.0
# p3's window voice gate is 3 frames x 30 ms = 90 ms of speech; p4
# frames are 40 ms, so the same span rounded up to the frame grid is
# ceil(90 / 40) = 3 frames (120 ms, slightly stricter, fewer false
# triggers on an energy-only gate).
WINDOW_VOICE_FRAMES = 3
WINDOW_BEEP_PATH = "/root/filler/beep.wav"
WINDOW_TIMEOUT_TEXT = "就到这里了，再见"
# p4 energy gate: the BT600 level (never Pi3's 1000).
WINDOW_RMS_THRESHOLD = EnergyStop.RMS_THRESHOLD


def _play_wav_bytes_via_file(data: bytes, play_file) -> None:
    """Play WAV bytes through the file player via a temp file.

    Only the synthesized goodbye needs this: the beep and the close
    pads are already files. The temp file is removed afterwards.
    """
    fd, tmp = tempfile.mkstemp(prefix="musegadget-window-", suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        play_file(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _play_close_pad(beep_path, play_file) -> bool:
    """Play one close pad (close_*.wav next to the beep file) if any.

    Returns True if a pad was played, False if none was found or the
    pad could not be played (the caller then falls back to the
    synthesized goodbye, so a missing pad never silences the window).
    Pads rotate so consecutive windows do not repeat the same phrase.
    """
    try:
        pads = sorted(Path(beep_path).parent.glob("close_*.wav"))
    except Exception:
        return False
    if not pads:
        return False
    idx = getattr(_play_close_pad, "_idx", 0) % len(pads)
    _play_close_pad._idx = idx + 1
    try:
        play_file(str(pads[idx]))
    except Exception:
        return False
    return True


def wait_window(hub, wake, wake_queue, button_events, stop_event,
                window_s: float = WINDOW_S, pause_s: float = WINDOW_PAUSE_S,
                beep_path: str = WINDOW_BEEP_PATH,
                voice_frames: int = WINDOW_VOICE_FRAMES,
                synth=_synthesize_with_fallback,
                aplay_device: str = DEFAULT_APLAY_DEVICE,
                play_file=None, play_wav_bytes=None) -> str | None:
    """Continuous-conversation window after a formal reply.

    Returns "wake", "hook", "pair", or "voice" for the first trigger,
    or None on timeout or stop. Timing: 1 s pause, then the beep, then
    a 5 s window. The beep and the timeout goodbye are played as
    files (play_prompt: a fresh aplay per sound, waited to the end),
    the shape the Pi3 loop uses, whose window works: the window clock
    starts only when the beep play returns, so a slow audio path
    delays the window instead of silently eating it. (The earlier
    streaming design started the clock when the frames left the
    queue, which is not when they become audible, and its fresh
    streaming aplay raced the previous round's teardown.) The voice
    gate counts only live frames after the beep: subscribe_record's
    snapshot (the 1 s pre-roll, holding the formal reply's tail and
    the beep) is ignored, so playback tail cannot trigger. Mute and
    hook-release in the window are ignored. On timeout a close pad
    plays (the synthesized goodbye is the fallback) and the caller
    returns to idle.
    The wake tap is armed for the whole window and disarmed on exit;
    recording disarms at its own boundary, as before.
    """
    if play_file is None:
        play_file = lambda path: play_prompt(path, aplay_device)  # noqa: E731
    if play_wav_bytes is None:
        play_wav_bytes = lambda data: _play_wav_bytes_via_file(data, play_file)  # noqa: E731
    if wake is not None:
        try:
            wake.arm()
        except Exception as exc:
            log.warning("window wake arm failed (%s)", type(exc).__name__)
    try:
        if stop_event is not None and stop_event.wait(pause_s):
            return None
        if Path(beep_path).is_file():
            started = time.monotonic()
            try:
                play_file(beep_path)
                log.info("window beep played in %.2fs",
                         time.monotonic() - started)
            except Exception as exc:
                log.warning("window beep skipped (%s)", type(exc).__name__)
        else:
            log.warning("window beep missing, skipping")
        record_queue = None
        if hub is not None:
            try:
                record_queue, _snap = hub.subscribe_record()
            except Exception as exc:
                log.warning("window voice gate unavailable (%s)", type(exc).__name__)
                record_queue = None
        try:
            deadline = time.monotonic() + window_s
            speech = 0
            while time.monotonic() < deadline:
                if stop_event is not None and stop_event.is_set():
                    return None
                while True:
                    event = next_button(button_events)
                    if event is None:
                        break
                    if event == "pair":
                        return "pair"
                    if event == "hook-press":
                        return "hook"
                    # mute and hook-release are ignored in the window.
                if wake_queue is not None:
                    try:
                        wake_queue.get_nowait()
                        return "wake"
                    except queue.Empty:
                        pass
                if record_queue is not None:
                    try:
                        frame = record_queue.get(timeout=0.02)
                    except queue.Empty:
                        continue
                    if frame_rms(frame) >= WINDOW_RMS_THRESHOLD:
                        speech += 1
                        if speech >= voice_frames:
                            return "voice"
                    else:
                        speech = 0
                else:
                    time.sleep(0.02)
        finally:
            if record_queue is not None:
                try:
                    hub.unsubscribe_record(record_queue)
                except Exception:
                    pass
        if stop_event is not None and stop_event.is_set():
            return None
        try:
            if not _play_close_pad(beep_path, play_file):
                play_wav_bytes(synth(WINDOW_TIMEOUT_TEXT))
            log.info("window timeout, goodbye played")
        except Exception as exc:
            log.warning("window goodbye skipped (%s)", type(exc).__name__)
        return None
    finally:
        if wake is not None:
            try:
                wake.disarm()
            except Exception as exc:
                log.warning("window wake disarm failed (%s)", type(exc).__name__)


async def _refresh_pairing_in_place(pairing: dict, node_id: str) -> int | None:
    """Rotate the device token pair in place and persist it.

    Returns the HTTP status of the refresh attempt (200 on success),
    or None on transport failure. Mirrors the Pi3 loop's settled
    rules (2026-10-31): the caller decides when to refresh; a 401
    here means the pairing is revoked and must not be retried in a
    loop. The pairing dict is mutated, so the run loop's copy stays
    current across rounds.
    """
    try:
        sdk = config.sdk_token()
    except Exception:
        sdk = None
    try:
        tokens, status = await asyncio.to_thread(
            muse_api.refresh_device_token,
            pairing["refresh_token"], node_id,
            muse_api.api_root(pairing.get("api_url_v2", "")), sdk)
    except Exception as exc:
        # A refresh attempt must never kill the round: the fetch
        # that follows decides (and its 401 path forces one retry).
        log.warning("token refresh errored (%s)", type(exc).__name__)
        return None
    if not tokens:
        log.warning("token refresh failed (status=%s)", status)
        return status
    pairing["access_token"] = tokens["access_token"]
    pairing["refresh_token"] = tokens["refresh_token"]
    pairing["access_token_saved_at"] = int(time.time())
    try:
        config.save_json(config.PAIRING_FILE, pairing)
    except Exception as exc:
        log.warning("pairing save after refresh failed (%s)",
                    type(exc).__name__)
    log.info("device token refreshed")
    return status


async def _fetch_vms_with_refresh(pairing: dict, node_id: str):
    """fetch_vms_with_status with the Pi3 loop's refresh rules:
    pre-refresh when the saved access token is over 3 h old (the
    device tokens live about a day; without this the loop dies on a
    401 mid-evening, as on 2026-11-02), and on a 401 force one
    refresh and retry the fetch once. Raises RuntimeError when the
    refresh itself is rejected with 401 (pairing revoked).
    """
    age = time.time() - pairing.get("access_token_saved_at", 0)
    if age > 3 * 3600:
        await _refresh_pairing_in_place(pairing, node_id)
    vms, status = await asyncio.to_thread(
        muse_api.fetch_vms_with_status,
        pairing["access_token"],
        muse_api.api_root(pairing.get("api_url_v2", "")),
    )
    if status == 401:
        log.warning("VM fetch 401, forcing token refresh once")
        rstatus = await _refresh_pairing_in_place(pairing, node_id)
        if rstatus == 401:
            raise RuntimeError("pairing revoked, need re-pair")
        if rstatus == 200:
            vms, status = await asyncio.to_thread(
                muse_api.fetch_vms_with_status,
                pairing["access_token"],
                muse_api.api_root(pairing.get("api_url_v2", "")),
            )
    return vms, status


async def run_round(
    pairing: dict,
    node_id: str,
    audio,
    button_events: queue.Queue,
    proxy: str | None,
    enable_filler: bool = True,
    wake_prompt: bool = False,
    wake=None,
    wake_queue: queue.Queue | None = None,
    vad=None,
) -> tuple[str, bool, str | None]:
    """Run one voice-note round. Returns (reason, pair_pending, reply_text).

    Record (VAD stop when a sidecar is given, energy fallback otherwise)
    waiting fillers during the reply wait -> speak the text reply
    through the TTS leg. A fresh LinkSession per round, registration
    on, the same shape the voice-note test proved. Button mute
    discards/abandons; button pair submits early / abandons and
    chains a fresh round. enable_filler False keeps the WAV-test
    output clean (no filler frames in the file). wake_prompt True
    starts one wake-pool cue concurrently with registration,
    before the record window opens (wake rounds only; pair rounds
    keep their own prompt, chained rounds keep it too). The ring
    keeps speech from the cue and registration window, so the user
    hears the cue at once without losing speech. Wake tap arming
    (wake object; None keeps the old behavior): recording disarmed;
    receipt playback, reply wait and formal reply playback armed;
    the wake cue itself stays disarmed. Interrupt (step 3): formal
    playback interrupts on a wake word or a hook press, reply wait
    interrupts on a wake word, both with reason "interrupted" and a
    chained round (pair_pending True); recording submits on a hook
    release exactly like a pair press (speech >= 0.3 s submits, less
    discards), a hook press during recording is ignored. Receipt
    playback and the close phrase never interrupt: words queued
    during receipt are drained before the reply wait, and run_loop
    drains what is left after the round. The continuous window
    (step 4) arms from run_loop after the formal reply. Failures log only
    their type, never their text (text may echo credentials or user
    data)."""
    import logging as _logging

    _logging.getLogger("websockets").setLevel(_logging.WARNING)
    # Filler setup first (moved up from after registration): the wake
    # cue must start before registration so the user hears it at once;
    # the ring keeps speech from the cue and registration window.
    filler = None
    pools = None
    if enable_filler and isinstance(audio, P4Audio):
        filler = load_filler_module()
        if filler is not None:
            try:
                pools = filler.load_pools()
            except Exception as exc:
                log.warning("filler pools unavailable (%s)", type(exc).__name__)
                pools = None
    filler_on = (
        pools is not None
        and bool(pools.get("waiting"))
        and bool(pools.get("receipt"))
    )
    cue_task = None
    if wake_prompt and filler is not None and pools is not None:
        cue_task = asyncio.create_task(
            play_pool_cue(filler, pools, "wake", audio, button_events)
        )

    async def _cancel_cue() -> None:
        nonlocal cue_task
        if cue_task is not None and not cue_task.done():
            cue_task.cancel()
            await asyncio.gather(cue_task, return_exceptions=True)
        cue_task = None

    def _arm() -> None:
        # Best-effort: a tap failure must never fail the round.
        if wake is not None:
            try:
                wake.arm()
            except Exception as exc:
                log.warning("wake arm failed (%s)", type(exc).__name__)

    def _disarm() -> None:
        if wake is not None:
            try:
                wake.disarm()
            except Exception as exc:
                log.warning("wake disarm failed (%s)", type(exc).__name__)

    def _aec_on() -> None:
        # Best-effort like arm/disarm: AEC must never fail the round.
        hub = getattr(audio, "_hub", None)
        if hub is not None:
            try:
                hub.enable_aec()
            except Exception as exc:
                log.warning("AEC enable failed (%s)", type(exc).__name__)

    def _aec_off() -> None:
        hub = getattr(audio, "_hub", None)
        if hub is not None:
            try:
                hub.disable_aec()
            except Exception as exc:
                log.warning("AEC disable failed (%s)", type(exc).__name__)

    try:
        vms, status = await _fetch_vms_with_refresh(pairing, node_id)
    except RuntimeError:
        log.warning("round failed: pairing revoked, need re-pair")
        await _cancel_cue()
        return "vm_lookup_failed", False, None
    except Exception as exc:
        log.warning("round failed at VM lookup (%s)", type(exc).__name__)
        await _cancel_cue()
        return "vm_lookup_failed", False, None
    if status != 200:
        log.warning("round failed: VM lookup status not 200")
        await _cancel_cue()
        return "vm_lookup_failed", False, None
    if len(vms) != 1:
        log.warning("round skipped: %d VMs (need exactly 1)", len(vms))
        await _cancel_cue()
        return "vm_count", False, None
    vm = vms[0]
    session = LinkSession(
        noise_host=pairing.get("noise_host") or DEFAULT_NOISE_HOST,
        vm_id=vm["vm_id"],
        vm_auth_token=vm["vm_auth_token"],
        app_id="hatch-web",
        register_device=True,
        device=DeviceDescription(node_id, "Muse voice", _version(), {}),
        run_command=lambda *unused: {"ok": False, "error": "unsupported command"},
        proxy=proxy,
    )
    stop = asyncio.Event()
    connection = asyncio.create_task(session.run(stop))
    try:
        await session.wait_transport_ready(35)
        # --- wake prompt (started before registration, awaited here) ---
        # The cue plays concurrently with registration; the ring keeps
        # speech from the cue and registration window, so the record
        # window still starts from the pre-roll. Best-effort like
        # receipt: a missing pool or file never fails the round.
        if cue_task is not None:
            wake_reason = await cue_task
            cue_task = None
            if wake_reason == "cancelled-mute":
                return "cancelled-mute", False, None
            if wake_reason == "cancelled-pair":
                return "cancelled-pair", True, None
            # ok / audio_failed / None: cue only, the round goes on.
        # --- record (wake tap disarmed) ---
        _disarm()
        stopper = VadStop() if vad is not None else EnergyStop()
        last_vad = False
        pcm = bytearray()
        submitted = False
        try:
            async for frame in audio.frames():
                pcm += frame
                early_submit = False
                while True:
                    event = next_button(button_events)
                    if event is None:
                        break
                    if event == "mute":
                        return "cancelled-mute", False, None
                    if event == "pair" or event == "hook-release":
                        # Toggle rule (same as Pi3): a press during
                        # recording is ignored; a release submits when
                        # enough speech is in, discards otherwise.
                        if stopper.speech >= EnergyStop.MIN_SPEECH_FRAMES:
                            early_submit = True
                        else:
                            return "discarded", False, None
                        break
                    # A hook press during recording is ignored.
                if early_submit:
                    submitted = True
                    break
                if vad is not None:
                    # VAD verdicts for this frame; a dead sidecar fails
                    # the record (audio_failed) instead of silently
                    # falling back mid-round. No verdict yet holds the
                    # last one: the sidecar lags at most a frame.
                    if not vad.alive():
                        raise OSError("vad sidecar stopped")
                    vad.feed_input(resample_pcm(frame, SAMPLE_RATE, 16000))
                    decision = vad.next_decision()
                    if decision is None:
                        decision = last_vad
                    else:
                        last_vad = decision
                    decision = stopper.feed(decision)
                else:
                    decision = stopper.feed(frame_rms(frame))
                if decision == "submit":
                    submitted = True
                    break
                if decision == "discarded":
                    if await play_failure_feedback(NOT_HEARD_TEXT, audio, button_events) == "cancelled-pair":
                        return "cancelled-pair", True, None
                    return "discarded", False, None
        except OSError as exc:
            log.warning("round failed at capture (%s)", type(exc).__name__)
            return "audio_failed", False, None
        if not submitted and stopper.speech < EnergyStop.MIN_SPEECH_FRAMES:
            if await play_failure_feedback(NOT_HEARD_TEXT, audio, button_events) == "cancelled-pair":
                return "cancelled-pair", True, None
            return "discarded", False, None
        # --- receipt filler (Pi3: right after speech ends) ---
        if filler_on:
            receipt_path = filler.pick(pools["receipt"])
            if receipt_path:
                try:
                    receipt_frames = wav_bytes_to_frames(Path(receipt_path).read_bytes())
                except Exception as exc:
                    log.warning("receipt filler skipped (%s)", type(exc).__name__)
                    receipt_frames = []
                if receipt_frames:
                    _arm()
                    receipt_reason = await play_wav_frames(receipt_frames, audio, button_events)
                    _disarm()
                    if receipt_reason == "cancelled-mute":
                        return "cancelled-mute", False, None
                    if receipt_reason == "cancelled-pair":
                        return "cancelled-pair", True, None
                    # ok / audio_failed: garnish only, the round goes on.
        # --- send as a voice note, collect the text reply ---
        wav_bytes = wrap_wav(bytes(pcm), SAMPLE_RATE)
        if len(wav_bytes) > 2 * 1024 * 1024:
            log.warning("round skipped: voice note over 2 MiB")
            return "too_long", False, None
        collector = ReplyCollector()

        async def collect() -> None:
            # ValueError-only skip (the 256 KiB cap shape): log, sleep
            # 1 s, resubscribe, at most MAX_SUBSCRIBE_SKIPS consecutive
            # times; past that the task ends and the round reports
            # failed. Other errors keep the old behavior (log and end).
            # A ValueError after the reply is already done is a normal
            # end, not a skip.
            skips = 0
            while True:
                try:
                    async for event in session.subscribe_chat():
                        collector.feed(event)
                        if collector.done:
                            return
                    return
                except ValueError as exc:
                    if collector.done:
                        return
                    skips += 1
                    if skips > MAX_SUBSCRIBE_SKIPS:
                        log.warning("subscribe skipped %d times, giving up", skips)
                        return
                    log.warning("subscribe skipped (%s), retrying in 1 s",
                                type(exc).__name__)
                    await asyncio.sleep(1)
                except Exception as exc:
                    log.warning("subscribe ended (%s)", type(exc).__name__)
                    return

        collect_task = asyncio.create_task(collect())
        await asyncio.sleep(0.5)
        try:
            result = await session.send_voice(wav_bytes)
        except Exception as exc:
            log.warning("round failed at send (%s)", type(exc).__name__)
            collect_task.cancel()
            await asyncio.gather(collect_task, return_exceptions=True)
            if await play_failure_feedback(NO_REPLY_TEXT, audio, button_events) == "cancelled-pair":
                return "cancelled-pair", True, None
            return "failed", False, None
        if not result.get("ok"):
            log.warning("round failed: voice note not acknowledged")
            collect_task.cancel()
            await asyncio.gather(collect_task, return_exceptions=True)
            if await play_failure_feedback(NO_REPLY_TEXT, audio, button_events) == "cancelled-pair":
                return "cancelled-pair", True, None
            return "failed", False, None
        loop = asyncio.get_running_loop()
        started = loop.time()
        wait_reason: tuple[str, bool] | None = None
        # Waiting-filler state (Pi3 timing: first at wait+10 s, then
        # every 10 s; killed before the formal reply plays).
        last_filler_t = loop.time()
        filler_frames: list[bytes] = []
        filler_idx = 0
        prev_wait: str | None = None
        # Pre-wait drain (Pi3 rule): words queued during receipt
        # (armed but never interrupted) must not trip the wait at
        # once; button residue from the send window goes too.
        drain(button_events)
        if wake_queue is not None:
            drain(wake_queue)
        _arm()  # reply wait armed; a word now interrupts (below)
        while not collector.done:
            while True:
                event = next_button(button_events)
                if event is None:
                    break
                if event == "mute":
                    wait_reason = ("cancelled-mute", False)
                    break
                if event == "pair":
                    wait_reason = ("cancelled-pair", True)
                    break
                # Hook edges do not abandon a wait in v1.
            if wait_reason is not None or collect_task.done():
                break
            # Wake-word interrupt: abandon the wait and chain a
            # fresh round. A reply that already arrived wins over
            # a queued word (result priority).
            if wake_queue is not None:
                try:
                    wake_queue.get_nowait()
                except queue.Empty:
                    pass
                else:
                    if collector.done or collect_task.done():
                        break
                    wait_reason = ("interrupted", True)
                    break
            if loop.time() - started > REPLY_TIMEOUT_S:
                break
            if filler_on:
                if filler_frames and filler_idx < len(filler_frames):
                    # Burst-feed (up to 3 frames per 0.1 s poll): aplay's
                    # own clock paces output; the queue absorbs bursts.
                    for _ in range(3):
                        if filler_idx >= len(filler_frames):
                            break
                        try:
                            audio.write(filler_frames[filler_idx])
                            filler_idx += 1
                        except OSError:
                            break  # backpressure: retry next poll
                elif filler_frames:
                    filler_frames = []
                    filler_idx = 0
                elif loop.time() - last_filler_t >= WAIT_FILLER_S:
                    picked = filler.pick(pools["waiting"], exclude=prev_wait)
                    if picked:
                        try:
                            filler_frames = wav_bytes_to_frames(Path(picked).read_bytes())
                            filler_idx = 0
                            last_filler_t = loop.time()
                            prev_wait = picked
                        except Exception as exc:
                            log.warning("waiting filler skipped (%s)", type(exc).__name__)
                            last_filler_t = loop.time()
            await asyncio.sleep(0.1)
        _disarm()  # reply wait over
        collect_task.cancel()
        await asyncio.gather(collect_task, return_exceptions=True)
        audio.flush()  # kill any waiting filler before what comes next
        if wait_reason is not None:
            # The close phrase stays disarmed: the round ends here.
            if wait_reason[0] == "cancelled-mute" and filler_on:
                close_files = sorted(glob.glob(os.path.join(FILLER_DIR, "close_*.wav")))
                if close_files:
                    try:
                        close_frames = wav_bytes_to_frames(Path(random.choice(close_files)).read_bytes())
                    except Exception as exc:
                        log.warning("close filler skipped (%s)", type(exc).__name__)
                        close_frames = []
                    if close_frames:
                        close_reason = await play_wav_frames(close_frames, audio, button_events)
                        if close_reason == "cancelled-pair":
                            return "cancelled-pair", True, None
            return wait_reason[0], wait_reason[1], None
        if collector.done and collector.text:
            log.info("reply received: %d chars", len(collector.text))
            _arm()
            _aec_on()
            speak_reason = await speak_text(
                collector.text, audio, button_events,
                wake_queue=wake_queue, hook_interrupt=True)
            _aec_off()
            _disarm()
            if speak_reason == "ok":
                return "ok", False, collector.text
            if speak_reason == "cancelled-pair":
                return "cancelled-pair", True, collector.text
            if speak_reason == "cancelled-mute":
                return "cancelled-mute", False, collector.text
            if speak_reason == "interrupted":
                return "interrupted", True, collector.text
            return speak_reason, False, collector.text
        if collector.done:
            if await play_failure_feedback(NOT_HEARD_TEXT, audio, button_events) == "cancelled-pair":
                return "cancelled-pair", True, None
            return "empty_reply", False, None
        # Wait over with no reply (failed stream or timeout): one phrase
        # for both outcomes; the original reason below is unchanged.
        if await play_failure_feedback(NO_REPLY_TEXT, audio, button_events) == "cancelled-pair":
            return "cancelled-pair", True, None
        if collect_task.done():
            return "failed", False, None
        return "reply_timeout", False, None
    except Exception as exc:
        log.warning("round failed (%s)", type(exc).__name__)
        await _cancel_cue()
        _disarm()
        if await play_failure_feedback(NO_REPLY_TEXT, audio, button_events) == "cancelled-pair":
            return "cancelled-pair", True, None
        return "failed", False, None
    finally:
        stop.set()
        try:
            await asyncio.wait_for(connection, 3)
        except (Exception, asyncio.CancelledError):
            connection.cancel()
            await asyncio.gather(connection, return_exceptions=True)


def _version() -> str:
    try:
        from musegadget import __version__ as version
    except Exception:
        return "0.1.0"
    return version


def run_loop(args: argparse.Namespace) -> int:
    """Wake -> round -> window -> idle, forever (systemd restarts on failure)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    stop_event = threading.Event()

    def on_signal(signum, frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    button_events: queue.Queue = queue.Queue()
    ButtonReader(args.fifo, button_events).start()
    wake_queue: queue.Queue = queue.Queue()
    hub = CaptureHub(args.arecord_device)
    hub.start()
    # One wake phase for the whole loop: the sherpa process lives
    # across rounds; only the tap is gated (arm/disarm) at boundaries.
    wake = WakePhase(args.model, args.wake_script, args.arecord_device, wake_queue,
                     hub=hub)
    # Pair prompts use one wake-pool phrase (step 5): the filler module
    # is loaded once here so its shuffle deck survives across prompts;
    # a missing module/pool falls back to the fixed prompt.
    pair_filler = load_filler_module()
    pair_pools = None
    if pair_filler is not None:
        try:
            pair_pools = pair_filler.load_pools()
        except Exception as exc:
            log.warning("pair filler pools unavailable (%s)", type(exc).__name__)
            pair_pools = None
    # One VAD phase for the whole loop (resident sidecar); a failed
    # start falls back to the energy stop, never blocks the loop.
    vad = None
    try:
        vad = VadPhase(args.vad_script, args.vad_model)
        vad.start()
    except Exception as exc:
        log.warning("VAD sidecar unavailable, energy fallback (%s)", type(exc).__name__)
        vad = None

    while not stop_event.is_set():
        pairing = config.load_json(config.PAIRING_FILE)
        if not pairing or not pairing.get("access_token"):
            log.warning("not paired; waiting for pairing")
            stop_event.wait(30)
            continue
        ident = identity.load_or_create()
        try:
            wake.start()
        except Exception as exc:
            log.warning("wake phase failed to start (%s)", type(exc).__name__)
            stop_event.wait(5)
            continue
        wake.arm()
        drain(wake_queue)
        trigger: str | None = None
        while trigger is None and not stop_event.is_set():
            try:
                event = button_events.get(timeout=0.5)
                if event == "pair":
                    trigger = "pair"
                elif event == "hook-press":
                    trigger = "hook"
                # Idle mute is ignored; hook-release is ignored.
            except queue.Empty:
                pass
            try:
                wake_queue.get_nowait()
                trigger = "wake"
            except queue.Empty:
                pass
            if not wake.alive():
                log.warning("wake phase died, restarting")
                wake.stop()
                break
        wake.disarm()
        if stop_event.is_set() or trigger is None:
            continue
        if trigger == "pair":
            play_pair_prompt(pair_filler, pair_pools, args.prompt, args.aplay_device)
        # Rounds, then the continuous window after a formal reply: a
        # window trigger starts another round (window rounds never play
        # the wake prompt), a window timeout returns to idle.
        window_round = False
        reason = None
        while True:
            # One round, plus a chained round per pair press during a round.
            # Reset: if this iteration runs zero rounds (audio unavailable),
            # the previous round's reason must not open another window.
            reason = None
            need_round = True
            first_round = True
            audio = None
            while need_round and not stop_event.is_set():
                drain(button_events)
                if vad is not None and not vad.alive():
                    try:
                        vad.start()
                    except Exception as exc:
                        log.warning("VAD sidecar restart failed, energy fallback (%s)",
                                    type(exc).__name__)
                        vad = None
                audio = None
                try:
                    audio = P4Audio(args.arecord_device, args.aplay_device, hub=hub)
                except Exception as exc:
                    log.warning("audio unavailable (%s)", type(exc).__name__)
                    break
                try:
                    reason, pair_pending, _reply = asyncio.run(
                        run_round(
                            pairing, ident.node_id, audio, button_events, args.proxy,
                            wake_prompt=(trigger == "wake" and first_round
                                         and not window_round),
                            wake=wake,
                            wake_queue=wake_queue,
                            vad=vad,
                        )
                    )
                finally:
                    audio.close()
                first_round = False
                log.info("round ended: %s", reason)
                need_round = pair_pending
                if need_round:
                    play_prompt(args.prompt, args.aplay_device)
            # Detections never acted on inside the round (receipt
            # playback is armed but not interrupted; the close phrase
            # plays disarmed): drop them so a stale word cannot start
            # a phantom round or window after this round ends.
            drain(wake_queue)
            if stop_event.is_set() or reason != "ok":
                break
            window_trigger = wait_window(
                hub, wake, wake_queue, button_events, stop_event,
                aplay_device=args.aplay_device,
            )
            if window_trigger is None:
                break
            trigger = window_trigger
            window_round = True
            if trigger == "pair":
                play_pair_prompt(pair_filler, pair_pools, args.prompt, args.aplay_device)
    wake.stop()
    if vad is not None:
        vad.stop()
    hub.stop()
    return 0


def run_wav_test(args: argparse.Namespace) -> int:
    """One round with WAV files instead of live audio: proves the
    voice-note send, the text reply, and the TTS leg without a
    microphone or speaker. Prints the reply text (diagnostic only)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    pairing = config.load_json(config.PAIRING_FILE)
    if not pairing or not pairing.get("access_token"):
        print("not paired")
        return 1
    ident = identity.load_or_create()
    button_events: queue.Queue = queue.Queue()
    try:
        audio = WavAudio(args.wav_in, args.wav_out)
    except Exception as exc:
        print(f"wav test failed ({type(exc).__name__})")
        return 1
    try:
        reason, _, reply = asyncio.run(
            run_round(
                pairing, ident.node_id, audio, button_events, args.proxy,
                enable_filler=False,
            )
        )
    finally:
        audio.close()
    if reply:
        print(f"reply text: {reply}")
    print(f"wav test ended: {reason}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="p4 car voice loop (rounds plus continuous window)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--wake-script", default=DEFAULT_WAKE_SCRIPT)
    parser.add_argument("--vad-script", default=DEFAULT_VAD_SCRIPT)
    parser.add_argument("--vad-model", default=DEFAULT_VAD_MODEL)
    parser.add_argument("--fifo", default=DEFAULT_FIFO)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--arecord-device", default=DEFAULT_ARECORD_DEVICE)
    parser.add_argument("--aplay-device", default=DEFAULT_APLAY_DEVICE)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--probe-audio", action="store_true")
    parser.add_argument("--wav-in", default=None)
    parser.add_argument("--wav-out", default=None)
    args = parser.parse_args(argv)
    if args.probe_audio:
        return probe_audio(args.arecord_device, args.aplay_device)
    if args.wav_in or args.wav_out:
        if not (args.wav_in and args.wav_out):
            parser.error("--wav-in and --wav-out must be given together")
        return run_wav_test(args)
    return run_loop(args)


if __name__ == "__main__":
    sys.exit(main())
