# Tests for the p4 car voice loop. Synthetic data only: no recordings,
# no credentials, no hardware, no network.

import asyncio
import queue
import struct
import threading
import time
import wave

from musegadget.link_client import DeviceDescription, LinkSession
from musegadget.p4_loop import (
    CaptureHub,
    EnergyStop,
    LinearAec,
    P4Audio,
    ReplyCollector,
    VadStop,
    WakePhase,
    build_ssml,
    cap_silences,
    frame_rms,
    load_filler_module,
    parse_button_line,
    play_pool_cue,
    play_wav_frames,
    read_tts_wav,
    resample_pcm,
    run_round,
    wait_window,
    MAX_SUBSCRIBE_SKIPS,
    NO_REPLY_TEXT,
    NOT_HEARD_TEXT,
    REPLY_TIMEOUT_S,
    play_pair_prompt,
    wav_bytes_to_frames,
    _CHILDREN,
    _bounded_round,
    _fetch_vms_with_refresh,
    _shutdown_watchdog,
    wrap_wav,
    WINDOW_TIMEOUT_TEXT,
    WINDOW_VOICE_FRAMES,
)


def _sine_pcm(n, rate, freq=440.0):
    import math

    return struct.pack(
        f"<{n}h",
        *[int(12000 * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)],
    )


def test_resample_lengths():
    frame_16k = _sine_pcm(640, 16000)
    up = resample_pcm(frame_16k, 16000, 24000)
    assert len(up) == 960 * 2
    frame_24k = _sine_pcm(960, 24000)
    down = resample_pcm(frame_24k, 24000, 16000)
    assert len(down) == 640 * 2
    assert resample_pcm(b"", 16000, 24000) == b""
    assert resample_pcm(frame_16k, 16000, 16000) == frame_16k


def test_resample_roundtrip_shape():
    original = _sine_pcm(640, 16000)
    back = resample_pcm(resample_pcm(original, 16000, 24000), 24000, 16000)
    assert len(back) == len(original)
    orig_samples = struct.unpack("<640h", original)
    back_samples = struct.unpack("<640h", back)
    peak_orig = max(abs(v) for v in orig_samples)
    peak_back = max(abs(v) for v in back_samples)
    assert peak_back > 0.5 * peak_orig
    assert all(-32768 <= v <= 32767 for v in back_samples)


def test_parse_button_line():
    assert parse_button_line("BUTTON_PAIR") == "pair"
    assert parse_button_line("BUTTON_MUTE") == "mute"
    assert parse_button_line("PRESS") == "hook-press"
    assert parse_button_line("RELEASE") == "hook-release"
    assert parse_button_line("  BUTTON_PAIR\n") == "pair"
    assert parse_button_line("1f04") is None
    assert parse_button_line("") is None


def test_link_session_proxy_stored():
    session = LinkSession(
        noise_host="noise.example",
        vm_id="vm-1",
        vm_auth_token="token-1",
        device=DeviceDescription("node-1", "Muse voice", "0.1.0", {}),
        run_command=lambda *unused: {"ok": False},
        proxy="http://127.0.0.1:7890",
    )
    assert session._proxy == "http://127.0.0.1:7890"
    direct = LinkSession(
        noise_host="noise.example",
        vm_id="vm-1",
        vm_auth_token="token-1",
        device=DeviceDescription("node-1", "Muse voice", "0.1.0", {}),
        run_command=lambda *unused: {"ok": False},
    )
    assert direct._proxy is None


def test_frame_rms():
    assert frame_rms(b"") == 0.0
    assert frame_rms(b"\x00\x00" * 960) == 0.0
    loud = struct.pack("<960h", *([10000] * 960))
    assert abs(frame_rms(loud) - 10000.0) < 1.0


def test_wrap_and_read_wav():
    pcm = _sine_pcm(960, 24000)
    data = wrap_wav(pcm, 24000)
    assert data[:4] == b"RIFF"
    back, rate = read_tts_wav(data)
    assert rate == 24000 and back == pcm


def test_read_tts_wav_stereo_keeps_left():
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setparams((2, 2, 22050, 0, "NONE", "not compressed"))
        frames = b"".join(
            struct.pack("<hh", 1000, -1000) for _ in range(882)
        )
        handle.writeframes(frames)
    back, rate = read_tts_wav(buffer.getvalue())
    assert rate == 22050
    samples = struct.unpack(f"<{len(back) // 2}h", back)
    assert len(samples) == 882 and all(v == 1000 for v in samples)


def test_energy_stop_no_speech_discards():
    stop = EnergyStop()
    for _ in range(EnergyStop.NO_SPEECH_FRAMES - 1):
        assert stop.feed(0.0) == "continue"
    assert stop.feed(0.0) == "discarded"


def test_energy_stop_silence_submits():
    stop = EnergyStop()
    for _ in range(EnergyStop.MIN_SPEECH_FRAMES):
        assert stop.feed(5000.0) == "continue"
    for _ in range(EnergyStop.SILENCE_STOP_FRAMES - 1):
        assert stop.feed(0.0) == "continue"
    assert stop.feed(0.0) == "submit"


def test_energy_stop_p4_threshold():
    # p4 BT600 level, measured 2026-11-02: speech frames peak at 362
    # RMS, silence peaks at 29; Pi3's 1000 threshold never trips on p4.
    stop = EnergyStop()
    for _ in range(EnergyStop.MIN_SPEECH_FRAMES):
        assert stop.feed(150.0) == "continue"
    for _ in range(EnergyStop.SILENCE_STOP_FRAMES - 1):
        assert stop.feed(0.0) == "continue"
    assert stop.feed(0.0) == "submit"


def test_energy_stop_cap_needs_speech():
    stop = EnergyStop()
    for _ in range(7):
        assert stop.feed(5000.0) == "continue"
    for _ in range(EnergyStop.MAX_FRAMES - 7 - 1):
        assert stop.feed(0.0) == "continue"
    assert stop.feed(0.0) == "discarded"


def test_build_ssml_escapes():
    ssml = build_ssml("a & b <c>").decode("utf-8")
    assert "a &amp; b &lt;c&gt;" in ssml
    assert "zh-CN-YunxiNeural" in ssml


def test_reply_collector():
    collector = ReplyCollector()
    collector.feed(
        {"event": "delta.text_append",
         "payload": {"role": "assistant", "text": "五", "message_id": "m1"}}
    )
    assert not collector.done and collector.text == "五"
    assert collector.message_id == "m1"
    collector.feed(
        {"event": "delta.text_append",
         "payload": {"role": "user", "text": "忽略"}}
    )
    assert collector.text == "五"
    collector.feed({"event": "delta.message_done", "payload": {"message_id": "m1"}})
    assert collector.done and collector.text == "五"
    whole = ReplyCollector()
    whole.feed(
        {"event": "message.assistant",
         "payload": {"text": "全文", "message_id": "m2"}}
    )
    assert whole.done and whole.text == "全文"


def test_reply_collector_empty_completion_does_not_complete():
    # WK-226: an agent turn that ends with an empty message must not
    # close the round; the collector swallows it and keeps waiting.
    collector = ReplyCollector()
    collector.feed({"event": "delta.message_done", "payload": {"message_id": "e1"}})
    assert not collector.done and collector.text == ""
    assert collector.empty_swallowed == 1 and collector.message_id is None
    # A duplicate completion event for the same empty message is not
    # counted twice.
    collector.feed({"event": "message.assistant", "payload": {"message_id": "e1"}})
    assert not collector.done and collector.empty_swallowed == 1
    # The substantive reply, arriving as a later message, completes.
    collector.feed(
        {"event": "delta.text_append",
         "payload": {"role": "assistant", "text": "实", "message_id": "m9"}}
    )
    collector.feed(
        {"event": "delta.text_append",
         "payload": {"role": "assistant", "text": "答", "message_id": "m9"}}
    )
    assert not collector.done and collector.message_id == "m9"
    collector.feed({"event": "delta.message_done", "payload": {"message_id": "m9"}})
    assert collector.done and collector.text == "实答"
    assert collector.empty_swallowed == 1


def test_reply_collector_empty_assistant_snapshot_then_full_one():
    collector = ReplyCollector()
    collector.feed(
        {"event": "message.assistant",
         "payload": {"text": "", "message_id": "e2"}}
    )
    assert not collector.done and collector.empty_swallowed == 1
    collector.feed(
        {"event": "message.assistant",
         "payload": {"display_text": "全文", "message_id": "m10"}}
    )
    assert collector.done and collector.text == "全文"


class _FakeSink:
    """Frame sink for play_wav_frames tests (no queue limit)."""

    def __init__(self, on_write=None):
        self.frames = []
        self.flushes = 0
        self.on_write = on_write

    def write(self, frame):
        self.frames.append(frame)
        if self.on_write:
            self.on_write(len(self.frames))

    def flush(self):
        self.flushes += 1


class _FailingSink:
    def write(self, frame):
        raise OSError("sink dead")

    def flush(self):
        pass


def test_wav_bytes_to_frames():
    pcm = _sine_pcm(882, 22050)
    frames = wav_bytes_to_frames(wrap_wav(pcm, 22050))
    assert len(frames) == 1 and len(frames[0]) == 960 * 2
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setparams((2, 2, 22050, 0, "NONE", "not compressed"))
        handle.writeframes(b"".join(struct.pack("<hh", 1000, -1000) for _ in range(882)))
    frames = wav_bytes_to_frames(buffer.getvalue())
    assert len(frames) == 1


def test_play_wav_frames_ok():
    sink = _FakeSink()
    frames = [b"\x00\x00" * 960] * 3
    result = asyncio.run(play_wav_frames(frames, sink, queue.Queue()))
    assert result == "ok" and len(sink.frames) == 3 and sink.flushes == 0


def test_play_wav_frames_mute():
    events: queue.Queue = queue.Queue()
    sink = _FakeSink(on_write=lambda n: events.put("mute") if n == 3 else None)
    frames = [b"\x00\x00" * 960] * 5
    result = asyncio.run(play_wav_frames(frames, sink, events))
    assert result == "cancelled-mute" and len(sink.frames) == 3 and sink.flushes == 1


def test_play_wav_frames_pair():
    events: queue.Queue = queue.Queue()
    sink = _FakeSink(on_write=lambda n: events.put("pair") if n == 2 else None)
    frames = [b"\x00\x00" * 960] * 5
    result = asyncio.run(play_wav_frames(frames, sink, events))
    assert result == "cancelled-pair" and len(sink.frames) == 2 and sink.flushes == 1


def test_play_wav_frames_audio_failed():
    result = asyncio.run(
        play_wav_frames([b"\x00\x00" * 960], _FailingSink(), queue.Queue())
    )
    assert result == "audio_failed"


def test_load_filler_module_missing():
    assert load_filler_module("/nonexistent/filler.py") is None


class _FakeAudio:
    def __init__(self):
        self.written = []

    def write(self, frame):
        self.written.append(frame)

    def flush(self):
        pass


class _FakeFiller:
    def pick(self, pool_files, exclude=None):
        return pool_files[0] if pool_files else None


def _write_test_wav(path, rate=22050, seconds=0.2):
    frames = _sine_pcm(int(rate * seconds), rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)


def test_play_pool_cue_plays_wake(tmp_path):
    wav_path = tmp_path / "wake.wav"
    _write_test_wav(wav_path)
    audio = _FakeAudio()
    reason = asyncio.run(
        play_pool_cue(_FakeFiller(), {"wake": [str(wav_path)]}, "wake", audio, queue.Queue())
    )
    assert reason == "ok"
    assert audio.written


def test_play_pool_cue_missing_pool_returns_none():
    audio = _FakeAudio()
    reason = asyncio.run(play_pool_cue(_FakeFiller(), {}, "wake", audio, queue.Queue()))
    assert reason is None
    assert audio.written == []


def _chunk(index):
    # One 40 ms chunk (1280 bytes), distinct in its first two bytes.
    return struct.pack("<H", index) + b"\x00" * 1278


class _ChunkStream:
    """Test-only byte stream fed chunk by chunk from another thread."""

    def __init__(self):
        self._queue = queue.Queue()
        self._closed = False
        self._buf = b""

    def feed(self, data):
        self._queue.put(data)

    def feed_eof(self):
        self._queue.put(None)

    def close_stream(self):
        self._closed = True

    def read(self, n):
        while len(self._buf) < n:
            if self._closed and self._queue.empty():
                break
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._closed:
                    break
                continue
            if item is None:
                self._closed = True
                break
            self._buf += item
        out, self._buf = self._buf[:n], self._buf[n:]
        return out


class _FakeProc:
    """Test-only arecord stand-in: stdout is a chunk stream."""

    def __init__(self, stream):
        self.stdout = stream
        self._terminated = False

    def poll(self):
        return 0 if self._terminated else None

    def terminate(self):
        self._terminated = True
        self.stdout.close_stream()

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.terminate()


def _wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _p4audio_with_hub(hub):
    # frames() needs no playback device, so build the adapter without
    # running its constructor (which would spawn aplay).
    audio = object.__new__(P4Audio)
    audio._hub = hub
    audio._closed = False
    audio._record_queue = None
    return audio


def test_capture_hub_ring_overflow_drops_oldest():
    stream = _ChunkStream()
    hub = CaptureHub("hw:FAKE,0", spawn=lambda: _FakeProc(stream))
    hub.start()
    try:
        for i in range(150):
            stream.feed(_chunk(i))
        assert _wait_until(
            lambda: len(hub.snapshot(125)) == 125
            and hub.snapshot(125)[-1] == _chunk(149)
        )
        assert hub.snapshot(125) == [_chunk(i) for i in range(25, 150)]
        assert hub.snapshot() == [_chunk(i) for i in range(125, 150)]
        assert hub.snapshot(0) == []
    finally:
        hub.stop()


def test_capture_hub_pre_roll_snapshot():
    stream = _ChunkStream()
    hub = CaptureHub("hw:FAKE,0", spawn=lambda: _FakeProc(stream))
    hub.start()
    try:
        for i in range(30):
            stream.feed(_chunk(i))
        assert _wait_until(
            lambda: len(hub.snapshot()) == 25 and hub.snapshot()[-1] == _chunk(29)
        )
        assert hub.snapshot() == [_chunk(i) for i in range(5, 30)]
        assert hub.snapshot(10) == [_chunk(i) for i in range(20, 30)]
    finally:
        hub.stop()
    short_stream = _ChunkStream()
    short_hub = CaptureHub("hw:FAKE,0", spawn=lambda: _FakeProc(short_stream))
    short_hub.start()
    try:
        for i in range(3):
            short_stream.feed(_chunk(i))
        assert _wait_until(lambda: short_hub.snapshot() == [_chunk(i) for i in range(3)])
    finally:
        short_hub.stop()


def test_capture_hub_wake_tap_arm_disarm():
    stream = _ChunkStream()
    hub = CaptureHub("hw:FAKE,0", spawn=lambda: _FakeProc(stream))
    hub.start()
    try:
        feed = hub.arm_wake()
        for i in range(3):
            stream.feed(_chunk(i))
        assert _wait_until(lambda: feed.qsize() >= 3)
        assert [feed.get_nowait() for _ in range(3)] == [_chunk(i) for i in range(3)]
        hub.disarm_wake()
        for i in range(3, 5):
            stream.feed(_chunk(i))
        time.sleep(0.3)
        assert feed.empty()
        feed2 = hub.arm_wake()
        assert feed2 is feed
        stream.feed(_chunk(5))
        assert _wait_until(lambda: feed.qsize() >= 1)
        assert feed.get_nowait() == _chunk(5)
    finally:
        hub.stop()


def test_capture_hub_relaunch_on_death():
    first = _ChunkStream()
    second = _ChunkStream()
    calls = []

    def spawn():
        calls.append(1)
        return _FakeProc(first if len(calls) == 1 else second)

    hub = CaptureHub("hw:FAKE,0", spawn=spawn)
    hub.start()
    try:
        for i in range(3):
            first.feed(_chunk(i))
        first.feed_eof()
        for i in range(100, 110):
            second.feed(_chunk(i))
        assert _wait_until(lambda: len(calls) >= 2)
        assert _wait_until(lambda: _chunk(100) in hub.snapshot(125))
    finally:
        hub.stop()


def test_p4audio_frames_pre_roll_then_live():
    stream = _ChunkStream()
    hub = CaptureHub("hw:FAKE,0", spawn=lambda: _FakeProc(stream))
    hub.start()
    try:
        for i in range(30):
            stream.feed(_chunk(i))
        assert _wait_until(
            lambda: len(hub.snapshot()) == 25 and hub.snapshot()[-1] == _chunk(29)
        )
        audio = _p4audio_with_hub(hub)

        async def collect():
            out = []
            gen = audio.frames()
            async for frame in gen:
                out.append(frame)
                if len(out) == 25:
                    stream.feed(_chunk(30))
                    stream.feed(_chunk(31))
                if len(out) >= 27:
                    break
            await gen.aclose()
            return out

        frames = asyncio.run(collect())
        assert frames[:25] == [
            resample_pcm(_chunk(i), 16000, 24000) for i in range(5, 30)
        ]
        assert frames[25:] == [
            resample_pcm(_chunk(i), 16000, 24000) for i in range(30, 32)
        ]
        assert len(hub._record_queues) == 0
    finally:
        hub.stop()


def test_p4audio_frames_stalled_capture_raises():
    def failing_spawn():
        raise RuntimeError("no device")

    hub = CaptureHub("hw:FAKE,0", spawn=failing_spawn)
    hub.start()
    try:
        audio = _p4audio_with_hub(hub)

        async def drain():
            try:
                async for _ in audio.frames():
                    pass
            except OSError:
                return True
            return False

        assert asyncio.run(drain())
    finally:
        hub.stop()


def test_capture_hub_kills_stalled_arecord_and_relaunches():
    first = _ChunkStream()
    second = _ChunkStream()
    calls = []

    def spawn():
        calls.append(1)
        return _FakeProc(first if len(calls) == 1 else second)

    hub = CaptureHub("hw:FAKE,0", spawn=spawn)
    hub.STALL_KILL_S = 0.4
    hub.start()
    try:
        first.feed(_chunk(0))
        first.feed(_chunk(1))
        assert _wait_until(lambda: len(hub.snapshot()) == 2)
        # First arecord stays alive but goes silent: the watchdog
        # must kill it (its stream closes, the reader relaunches)
        # instead of leaving every round to starve.
        assert _wait_until(lambda: len(calls) >= 2)
        second.feed(_chunk(50))
        assert _wait_until(lambda: _chunk(50) in hub.snapshot(125))
    finally:
        hub.stop()


class _WedgedStdin:
    """Playback stdin whose write blocks until the proc is killed."""

    def __init__(self):
        self.released = threading.Event()

    def write(self, data):
        self.released.wait(10)
        raise BrokenPipeError("wedged pipe released by the kill")

    def flush(self):
        pass


class _WedgedProc:
    pid = 4242

    def __init__(self):
        self.stdin = _WedgedStdin()
        self.killed = False
        self._dead = False

    def poll(self):
        return 0 if self._dead else None

    def kill(self):
        self.killed = True
        self._dead = True
        self.stdin.released.set()

    def terminate(self):
        self.kill()

    def wait(self, timeout=None):
        return 0


def _bare_p4audio(proc):
    audio = object.__new__(P4Audio)
    audio._aplay = proc
    audio._playback = queue.Queue(maxsize=8)
    audio._closed = False
    audio._playback_dead = False
    audio._write_in_flight = False
    audio._write_started_t = None
    audio._record_queue = None
    audio._hub = None
    audio._hub_owned = False
    return audio


def test_p4audio_playback_watchdog_kills_wedged_aplay():
    proc = _WedgedProc()
    audio = _bare_p4audio(proc)
    audio.PLAYBACK_STALL_S = 0.3
    audio._playback_thread = threading.Thread(
        target=audio._playback_loop, daemon=True)
    audio._playback_thread.start()
    watchdog = threading.Thread(
        target=audio._playback_watchdog_loop, daemon=True)
    watchdog.start()
    try:
        audio._playback.put(b"\x00" * 1920)
        assert _wait_until(lambda: proc.killed)
        assert _wait_until(lambda: audio._playback_dead)
    finally:
        audio._closed = True
        watchdog.join(timeout=2)
        audio._playback_thread.join(timeout=2)


class _BlockingCloseStdin:
    """stdin whose close() blocks until the proc is killed."""

    def __init__(self, proc):
        self._proc = proc

    def close(self):
        self._proc.close_released.wait(10)

    def write(self, data):
        return len(data)

    def flush(self):
        pass


class _StubbornProc:
    pid = 4343

    def __init__(self):
        self.close_released = threading.Event()
        self.stdin = _BlockingCloseStdin(self)
        self.terminated = False
        self.killed = False
        self._dead = False

    def poll(self):
        return 0 if self._dead else None

    def wait(self, timeout=None):
        if self._dead:
            return 0
        import subprocess
        raise subprocess.TimeoutExpired("fake-aplay", timeout)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self._dead = True
        self.close_released.set()


def test_p4audio_close_bounded_when_stdin_close_blocks():
    proc = _StubbornProc()
    audio = _bare_p4audio(proc)
    audio.CLOSE_STDIN_JOIN_S = 0.2
    audio.CLOSE_WAIT_S = 0.2
    started = time.monotonic()
    audio.close()
    elapsed = time.monotonic() - started
    # The full ladder is 0.2 (stdin join) + 0.2 (exit wait) + 0.2
    # (terminate wait) plus slack: far below the unbounded hang this
    # guards against.
    assert elapsed < 3.0
    assert proc.terminated and proc.killed


def test_shutdown_watchdog_kills_children_then_exits():
    proc = _WedgedProc()
    _CHILDREN.register(proc)
    exits = []
    stop_event = threading.Event()
    stop_event.set()
    try:
        _shutdown_watchdog(stop_event, grace_kill_s=0.3, grace_exit_s=0.3,
                           exit_fn=exits.append)
    finally:
        _CHILDREN.unregister(proc)
    assert proc.killed
    assert exits == [3]


def test_wake_phase_requires_hub():
    phase = WakePhase("model", "script", "hw:FAKE,0", queue.Queue())
    try:
        phase.start()
    except RuntimeError:
        return
    phase.stop()
    raise AssertionError("WakePhase without a hub must refuse to start")


_FAKE_SHERPA = (
    "#!/usr/bin/env python3\n"
    "import sys\n"
    "data = sys.stdin.buffer.read(2560)\n"
    "if len(data) >= 2560:\n"
    "    print('WAKE WORD DETECTED', flush=True)\n"
    "sys.stdin.buffer.read()\n"
)


def test_wake_phase_feeds_from_hub_tap(tmp_path):
    script = tmp_path / "fake-sherpa.py"
    script.write_text(_FAKE_SHERPA)
    script.chmod(0o755)
    stream = _ChunkStream()
    hub = CaptureHub("hw:FAKE,0", spawn=lambda: _FakeProc(stream))
    hub.start()
    try:
        # Pre-arm audio must not reach the wake tap.
        for i in range(2):
            stream.feed(_chunk(i))
        wake_queue = queue.Queue()
        phase = WakePhase("model", str(script), "hw:FAKE,0", wake_queue, hub=hub)
        phase.start()
        try:
            assert phase.alive()
            time.sleep(0.3)
            assert wake_queue.empty()
            for i in range(2, 5):
                stream.feed(_chunk(i))
            assert wake_queue.get(timeout=5) == "wake"
        finally:
            phase.stop()
        assert not phase.alive()
        assert not hub.wake_armed
    finally:
        hub.stop()


class _FakeRoundAudio(P4Audio):
    """run_round test audio: synthetic record frames, timed writes."""

    def __init__(self):
        self.written = []
        self.write_times = []
        self._closed = False

    async def frames(self):
        loud = struct.pack("<960h", *([5000] * 960))
        for _ in range(8):
            yield loud
        for _ in range(60):
            yield b"\x00\x00" * 960

    def write(self, frame):
        self.written.append(frame)
        self.write_times.append(time.monotonic())

    def flush(self):
        pass

    def close(self):
        self._closed = True


class _FakeFillerModule:
    def __init__(self, wav_path):
        self._wav = wav_path

    def load_pools(self):
        return {"wake": [self._wav], "waiting": [self._wav], "receipt": [self._wav]}

    def pick(self, pool_files, exclude=None):
        return pool_files[0] if pool_files else None


class _FakeApi:
    @staticmethod
    def fetch_vms_with_status(token, root):
        return ([{"vm_id": "v1", "vm_auth_token": "t1"}], 200)

    @staticmethod
    def api_root(root):
        return ""


class _FakeSession:
    last = None

    def __init__(self, *args, **kwargs):
        self.reg_done_t = None
        type(self).last = self

    async def run(self, stop):
        await stop.wait()

    async def wait_transport_ready(self, timeout):
        await asyncio.sleep(0.5)
        self.reg_done_t = time.monotonic()

    async def subscribe_chat(self):
        yield {"event": "delta.message_done",
               "payload": {"role": "assistant", "text": "五", "message_id": "m1"}}

    async def send_voice(self, wav_bytes):
        return {"ok": True}


def test_run_round_wake_cue_before_registration(tmp_path, monkeypatch):
    # The wake cue must start before registration finishes: the first
    # cue frame is written before the fake 0.5 s registration returns.
    wav_path = tmp_path / "wake.wav"
    _write_test_wav(wav_path)
    audio = _FakeRoundAudio()
    monkeypatch.setattr("musegadget.p4_loop.muse_api", _FakeApi)
    monkeypatch.setattr("musegadget.p4_loop.LinkSession", _FakeSession)
    monkeypatch.setattr(
        "musegadget.p4_loop.load_filler_module",
        lambda *args, **kwargs: _FakeFillerModule(str(wav_path)),
    )
    monkeypatch.setattr(
        "musegadget.p4_loop._synthesize_with_fallback",
        lambda text: wrap_wav(_sine_pcm(4410, 22050), 22050),
    )
    reason, pair_pending, text = asyncio.run(
        run_round(
            {"access_token": "x", "api_url_v2": ""},
            "node1", audio, queue.Queue(), None, wake_prompt=True,
        )
    )
    assert reason == "ok" and pair_pending is False and text == "五"
    session = _FakeSession.last
    assert session is not None and session.reg_done_t is not None
    assert audio.write_times and audio.write_times[0] < session.reg_done_t


_FAKE_SHERPA_LOOP = (
    "#!/usr/bin/env python3\n"
    "import sys\n"
    "while True:\n"
    "    data = sys.stdin.buffer.read(2560)\n"
    "    if not data:\n"
    "        break\n"
    "    if len(data) >= 2560:\n"
    "        print('WAKE WORD DETECTED', flush=True)\n"
)


def test_wake_phase_arm_disarm_survives_rounds(tmp_path):
    # Round boundaries must only gate the tap: the sherpa process and
    # its writer thread survive disarm, and detection resumes on re-arm.
    script = tmp_path / "fake-sherpa-loop.py"
    script.write_text(_FAKE_SHERPA_LOOP)
    script.chmod(0o755)
    stream = _ChunkStream()
    hub = CaptureHub("hw:FAKE,0", spawn=lambda: _FakeProc(stream))
    hub.start()
    try:
        wake_queue = queue.Queue()
        phase = WakePhase("model", str(script), "hw:FAKE,0", wake_queue, hub=hub)
        phase.start()
        try:
            assert phase.alive()
            proc = phase._sherpa
            phase.start()
            assert phase._sherpa is proc
            for i in range(2):
                stream.feed(_chunk(i))
            assert wake_queue.get(timeout=5) == "wake"
            phase.disarm()
            assert phase.alive()
            assert phase._sherpa is proc
            assert phase._writer is not None and phase._writer.is_alive()
            for i in range(2, 6):
                stream.feed(_chunk(i))
            time.sleep(0.3)
            assert wake_queue.empty()
            phase.arm()
            for i in range(6, 8):
                stream.feed(_chunk(i))
            assert wake_queue.get(timeout=5) == "wake"
            assert phase._sherpa is proc
            assert phase.alive()
        finally:
            phase.stop()
        assert not phase.alive()
        assert not hub.wake_armed
    finally:
        hub.stop()


class _FakeWake:
    """Recording wake tap for run_round arming tests."""

    def __init__(self):
        self.calls = []

    def arm(self):
        self.calls.append("arm")

    def disarm(self):
        self.calls.append("disarm")


class _FakeSessionDelayed(_FakeSession):
    async def subscribe_chat(self):
        await asyncio.sleep(0.3)
        yield {"event": "delta.message_done",
               "payload": {"role": "assistant", "text": "五", "message_id": "m1"}}


def test_run_round_wake_arming_brackets(tmp_path, monkeypatch):
    # Recording disarmed; receipt, reply wait and formal playback
    # armed, each bracket closed again before the round moves on.
    wav_path = tmp_path / "wake.wav"
    _write_test_wav(wav_path)
    audio = _FakeRoundAudio()
    wake = _FakeWake()
    monkeypatch.setattr("musegadget.p4_loop.muse_api", _FakeApi)
    monkeypatch.setattr("musegadget.p4_loop.LinkSession", _FakeSessionDelayed)
    monkeypatch.setattr(
        "musegadget.p4_loop.load_filler_module",
        lambda *args, **kwargs: _FakeFillerModule(str(wav_path)),
    )
    monkeypatch.setattr(
        "musegadget.p4_loop._synthesize_with_fallback",
        lambda text: wrap_wav(_sine_pcm(4410, 22050), 22050),
    )
    reason, pair_pending, text = asyncio.run(
        run_round(
            {"access_token": "x", "api_url_v2": ""},
            "node1", audio, queue.Queue(), None, wake_prompt=False, wake=wake,
        )
    )
    assert reason == "ok" and pair_pending is False and text == "五"
    assert wake.calls == ["disarm", "arm", "disarm", "arm", "disarm", "arm", "disarm"]


def test_play_wav_frames_wake_interrupt():
    sink = _FakeSink()
    frames = [b"\x00\x00" * 960] * 5
    wake_queue: queue.Queue = queue.Queue()
    wake_queue.put("wake")
    result = asyncio.run(
        play_wav_frames(frames, sink, queue.Queue(), wake_queue=wake_queue)
    )
    assert result == "interrupted" and len(sink.frames) == 0 and sink.flushes == 1


def test_play_wav_frames_wake_interrupt_mid_playback():
    wake_queue: queue.Queue = queue.Queue()
    sink = _FakeSink(
        on_write=lambda n: wake_queue.put("wake") if n == 2 else None)
    frames = [b"\x00\x00" * 960] * 5
    result = asyncio.run(
        play_wav_frames(frames, sink, queue.Queue(), wake_queue=wake_queue)
    )
    assert result == "interrupted" and len(sink.frames) == 2 and sink.flushes == 1


def test_play_wav_frames_hook_press_interrupt():
    events: queue.Queue = queue.Queue()
    events.put("hook-press")
    sink = _FakeSink()
    frames = [b"\x00\x00" * 960] * 5
    result = asyncio.run(
        play_wav_frames(frames, sink, events, hook_interrupt=True)
    )
    assert result == "interrupted" and len(sink.frames) == 0 and sink.flushes == 1


def test_play_wav_frames_hook_press_ignored_without_flag():
    # Receipt, close phrase and wake cue call without the flag: a hook
    # press must not interrupt them.
    events: queue.Queue = queue.Queue()
    events.put("hook-press")
    sink = _FakeSink()
    frames = [b"\x00\x00" * 960] * 5
    result = asyncio.run(play_wav_frames(frames, sink, events))
    assert result == "ok" and len(sink.frames) == 5 and sink.flushes == 0


def test_play_wav_frames_mute_beats_wake():
    events: queue.Queue = queue.Queue()
    events.put("mute")
    wake_queue: queue.Queue = queue.Queue()
    wake_queue.put("wake")
    sink = _FakeSink()
    frames = [b"\x00\x00" * 960] * 5
    result = asyncio.run(
        play_wav_frames(frames, sink, events, wake_queue=wake_queue,
                        hook_interrupt=True)
    )
    assert result == "cancelled-mute" and len(sink.frames) == 0


class _DelayedFakeAudio(_FakeRoundAudio):
    """Record frames paced 0.02 s apart so button injections land mid-record."""

    async def frames(self):
        loud = struct.pack("<960h", *([5000] * 960))
        for _ in range(8):
            yield loud
            await asyncio.sleep(0.02)
        for _ in range(60):
            yield b"\x00\x00" * 960
            await asyncio.sleep(0.02)


class _FakeSessionHung(_FakeSession):
    async def subscribe_chat(self):
        await asyncio.sleep(3600)
        yield {}


async def _run_with_put(target, item, delay, run_coro):
    async def _put():
        await asyncio.sleep(delay)
        target.put(item)

    task = asyncio.create_task(_put())
    try:
        return await run_coro
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _run_round_step3(tmp_path, monkeypatch, session_cls, audio, button_events,
                     wake_queue, synth_seconds=0.2):
    wav_path = tmp_path / "wake.wav"
    _write_test_wav(wav_path)
    monkeypatch.setattr("musegadget.p4_loop.muse_api", _FakeApi)
    monkeypatch.setattr("musegadget.p4_loop.LinkSession", session_cls)
    monkeypatch.setattr(
        "musegadget.p4_loop.load_filler_module",
        lambda *args, **kwargs: _FakeFillerModule(str(wav_path)),
    )
    samples = int(22050 * synth_seconds)
    monkeypatch.setattr(
        "musegadget.p4_loop._synthesize_with_fallback",
        lambda text: wrap_wav(_sine_pcm(samples, 22050), 22050),
    )
    return run_round(
        {"access_token": "x", "api_url_v2": ""},
        "node1", audio, button_events, None,
        wake_prompt=False, wake=_FakeWake(), wake_queue=wake_queue,
    )


def test_run_round_wait_wake_interrupt(tmp_path, monkeypatch):
    # Word injected during the reply wait abandons the wait and chains.
    wake_queue: queue.Queue = queue.Queue()
    coro = _run_round_step3(
        tmp_path, monkeypatch, _FakeSessionHung, _FakeRoundAudio(),
        queue.Queue(), wake_queue)
    reason, pair_pending, text = asyncio.run(
        _run_with_put(wake_queue, "wake", 1.5, coro))
    assert reason == "interrupted" and pair_pending is True and text is None


def test_run_round_wait_receipt_word_drained(tmp_path, monkeypatch):
    # A word queued before the round (receipt residue) must not trip
    # the wait: the wait ends on the injected mute instead.
    wake_queue: queue.Queue = queue.Queue()
    wake_queue.put("wake")
    button_events: queue.Queue = queue.Queue()
    coro = _run_round_step3(
        tmp_path, monkeypatch, _FakeSessionHung, _FakeRoundAudio(),
        button_events, wake_queue)
    reason, pair_pending, text = asyncio.run(
        _run_with_put(button_events, "mute", 1.5, coro))
    assert reason == "cancelled-mute" and pair_pending is False and text is None


def test_run_round_record_hook_release_submits(tmp_path, monkeypatch):
    # Release after 0.3 s of speech submits (same rule as pair).
    button_events: queue.Queue = queue.Queue()
    coro = _run_round_step3(
        tmp_path, monkeypatch, _FakeSession, _DelayedFakeAudio(),
        button_events, queue.Queue())
    reason, pair_pending, text = asyncio.run(
        _run_with_put(button_events, "hook-release", 1.1, coro))
    assert reason == "ok" and pair_pending is False and text == "五"


def test_run_round_record_hook_release_early_discards(tmp_path, monkeypatch):
    # Release before 0.3 s of speech discards (same rule as pair).
    button_events: queue.Queue = queue.Queue()
    coro = _run_round_step3(
        tmp_path, monkeypatch, _FakeSession, _DelayedFakeAudio(),
        button_events, queue.Queue())
    reason, pair_pending, text = asyncio.run(
        _run_with_put(button_events, "hook-release", 0.6, coro))
    assert reason == "discarded" and pair_pending is False and text is None


def test_run_round_record_hook_press_ignored(tmp_path, monkeypatch):
    # A press during recording is ignored; the round runs to silence stop.
    button_events: queue.Queue = queue.Queue()
    coro = _run_round_step3(
        tmp_path, monkeypatch, _FakeSession, _DelayedFakeAudio(),
        button_events, queue.Queue())
    reason, pair_pending, text = asyncio.run(
        _run_with_put(button_events, "hook-press", 1.1, coro))
    assert reason == "ok" and pair_pending is False and text == "五"


def test_run_round_speak_wake_interrupt(tmp_path, monkeypatch):
    # Word injected during formal playback stops it and chains.
    wake_queue: queue.Queue = queue.Queue()
    coro = _run_round_step3(
        tmp_path, monkeypatch, _FakeSession, _FakeRoundAudio(),
        queue.Queue(), wake_queue, synth_seconds=1.0)
    reason, pair_pending, text = asyncio.run(
        _run_with_put(wake_queue, "wake", 1.5, coro))
    assert reason == "interrupted" and pair_pending is True and text == "五"


def test_run_round_speak_hook_interrupt(tmp_path, monkeypatch):
    # Hook press during formal playback stops it and chains.
    button_events: queue.Queue = queue.Queue()
    coro = _run_round_step3(
        tmp_path, monkeypatch, _FakeSession, _FakeRoundAudio(),
        button_events, queue.Queue(), synth_seconds=1.0)
    reason, pair_pending, text = asyncio.run(
        _run_with_put(button_events, "hook-press", 1.5, coro))
    assert reason == "interrupted" and pair_pending is True and text == "五"


# --- step 4: continuous window ---

def _window_frame(loud: bool) -> bytes:
    # One 40 ms 16 kHz frame: loud RMS 5000 (>> gate 100), silence 0.
    return struct.pack("<640h", *([5000] * 640)) if loud else b"\x00\x00" * 640


class _FakeWindowWake:
    def __init__(self):
        self.arms = 0
        self.disarms = 0

    def arm(self):
        self.arms += 1

    def disarm(self):
        self.disarms += 1


class _FakeWindowHub:
    def __init__(self, snap=None, frames=None):
        self._snap = list(snap or [])
        self.record_queue: queue.Queue = queue.Queue()
        for frame in (frames or []):
            self.record_queue.put(frame)
        self.subscribed = 0
        self.unsubscribed = 0

    def subscribe_record(self):
        self.subscribed += 1
        return self.record_queue, list(self._snap)

    def unsubscribe_record(self, q):
        self.unsubscribed += 1


def _window_kwargs(**overrides):
    kwargs = dict(window_s=0.3, pause_s=0, beep_path="/nonexistent-beep.wav",
                  synth=lambda text: wrap_wav(_sine_pcm(4410, 22050), 22050))
    kwargs.update(overrides)
    return kwargs


class _FakePlayFile:
    """Window file player fake: records the paths it was asked to play."""

    def __init__(self):
        self.played: list[str] = []

    def __call__(self, path):
        self.played.append(str(path))


class _FakePlayWavBytes:
    """Window WAV-bytes player fake (the synthesized goodbye path)."""

    def __init__(self):
        self.played: list[bytes] = []

    def __call__(self, data):
        self.played.append(bytes(data))


def test_wait_window_voice_trigger():
    wake = _FakeWindowWake()
    hub = _FakeWindowHub(frames=[_window_frame(False),
                                 _window_frame(True), _window_frame(True),
                                 _window_frame(True)])
    result = wait_window(hub, wake, queue.Queue(), queue.Queue(),
                         threading.Event(), play_file=_FakePlayFile(),
                         **_window_kwargs())
    assert result == "voice"
    assert wake.arms == 1 and wake.disarms == 1
    assert hub.subscribed == 1 and hub.unsubscribed == 1


def test_wait_window_voice_gate_needs_consecutive_frames():
    # Two loud frames are not enough (gate is WINDOW_VOICE_FRAMES);
    # a silence resets the run.
    assert WINDOW_VOICE_FRAMES == 3
    hub = _FakeWindowHub(frames=[_window_frame(True), _window_frame(True)])
    result = wait_window(hub, _FakeWindowWake(), queue.Queue(), queue.Queue(),
                         threading.Event(), play_file=_FakePlayFile(),
                         **_window_kwargs())
    assert result is None
    hub = _FakeWindowHub(frames=[_window_frame(True), _window_frame(False),
                                 _window_frame(True), _window_frame(True)])
    result = wait_window(hub, _FakeWindowWake(), queue.Queue(), queue.Queue(),
                         threading.Event(), play_file=_FakePlayFile(),
                         **_window_kwargs())
    assert result is None


def test_wait_window_snapshot_does_not_trigger():
    # A loud 1 s pre-roll snapshot alone must not trigger: only live
    # frames after the beep count, so a formal reply's tail is safe.
    hub = _FakeWindowHub(snap=[_window_frame(True)] * 10)
    result = wait_window(hub, _FakeWindowWake(), queue.Queue(), queue.Queue(),
                         threading.Event(), play_file=_FakePlayFile(),
                         **_window_kwargs())
    assert result is None


def test_wait_window_wake_trigger():
    wake_queue: queue.Queue = queue.Queue()
    wake_queue.put("wake")
    result = wait_window(_FakeWindowHub(), _FakeWindowWake(), wake_queue, queue.Queue(),
                         threading.Event(), play_file=_FakePlayFile(),
                         **_window_kwargs())
    assert result == "wake"


def test_wait_window_pair_and_hook_triggers():
    buttons: queue.Queue = queue.Queue()
    buttons.put("pair")
    result = wait_window(_FakeWindowHub(), _FakeWindowWake(), queue.Queue(), buttons,
                         threading.Event(), play_file=_FakePlayFile(),
                         **_window_kwargs())
    assert result == "pair"
    buttons.put("hook-press")
    result = wait_window(_FakeWindowHub(), _FakeWindowWake(), queue.Queue(), buttons,
                         threading.Event(), play_file=_FakePlayFile(),
                         **_window_kwargs())
    assert result == "hook"


def test_wait_window_mute_ignored_then_timeout_says_goodbye():
    buttons: queue.Queue = queue.Queue()
    buttons.put("mute")
    said = []
    play_bytes = _FakePlayWavBytes()
    result = wait_window(_FakeWindowHub(), _FakeWindowWake(), queue.Queue(), buttons,
                         threading.Event(), play_file=_FakePlayFile(),
                         play_wav_bytes=play_bytes,
                         **_window_kwargs(synth=lambda text: said.append(text) or wrap_wav(_sine_pcm(4410, 22050), 22050)))
    assert result is None
    assert said == [WINDOW_TIMEOUT_TEXT]
    assert len(play_bytes.played) == 1


def test_wait_window_beep_plays_before_window(tmp_path):
    wav_path = tmp_path / "beep.wav"
    _write_test_wav(wav_path)
    play = _FakePlayFile()
    wake_queue: queue.Queue = queue.Queue()
    wake_queue.put("wake")
    result = wait_window(_FakeWindowHub(), _FakeWindowWake(), wake_queue, queue.Queue(),
                         threading.Event(), play_file=play,
                         **_window_kwargs(beep_path=str(wav_path)))
    assert result == "wake"
    assert play.played == [str(wav_path)]


def test_wait_window_beep_plays_before_gate_opens(tmp_path):
    # The gate subscribes only after the beep play has returned, so a
    # slow audio path delays the window instead of eating it.
    wav_path = tmp_path / "beep.wav"
    _write_test_wav(wav_path)
    order: list[str] = []
    hub = _FakeWindowHub()
    orig_subscribe = hub.subscribe_record

    def subscribe():
        order.append("gate")
        return orig_subscribe()

    hub.subscribe_record = subscribe

    def play_file(path):
        order.append("beep")

    wake_queue: queue.Queue = queue.Queue()
    wake_queue.put("wake")
    result = wait_window(hub, _FakeWindowWake(), wake_queue, queue.Queue(),
                         threading.Event(), play_file=play_file,
                         **_window_kwargs(beep_path=str(wav_path)))
    assert result == "wake"
    assert order == ["beep", "gate"]


def test_wait_window_no_player_still_triggers():
    wake_queue: queue.Queue = queue.Queue()
    wake_queue.put("wake")
    result = wait_window(_FakeWindowHub(), _FakeWindowWake(), wake_queue, queue.Queue(),
                         threading.Event(), **_window_kwargs())
    assert result == "wake"


def test_wait_window_stop_returns_none_without_goodbye():
    play = _FakePlayFile()
    play_bytes = _FakePlayWavBytes()
    stop = threading.Event()
    stop.set()
    said = []
    result = wait_window(_FakeWindowHub(), _FakeWindowWake(), queue.Queue(), queue.Queue(),
                         stop, play_file=play, play_wav_bytes=play_bytes,
                         **_window_kwargs(synth=lambda text: said.append(text) or b""))
    assert result is None
    assert said == []
    assert play.played == [] and play_bytes.played == []


# --- token refresh (2026-11-03, ported from the Pi3 loop) ---

class _FakeRefreshApi:
    """fetch/refresh fake with scripted fetch results."""

    def __init__(self, fetch_results, refresh_result):
        self._fetch_results = list(fetch_results)
        self._refresh_result = refresh_result
        self.fetch_calls = 0
        self.refresh_calls = 0

    def fetch_vms_with_status(self, token, root):
        self.fetch_calls += 1
        return self._fetch_results.pop(0)

    def refresh_device_token(self, refresh_token, device_id, root, sdk_token):
        self.refresh_calls += 1
        return self._refresh_result

    @staticmethod
    def api_root(root):
        return ""


class _FakeRefreshConfig:
    PAIRING_FILE = "pairing.json"

    def __init__(self):
        self.saved = []

    def sdk_token(self):
        return None

    def save_json(self, name, data):
        self.saved.append((name, dict(data)))


def _refresh_pairing(saved_at):
    return {"access_token": "old-access", "refresh_token": "old-refresh",
            "access_token_saved_at": saved_at}


def _patch_refresh(monkeypatch, api):
    config_fake = _FakeRefreshConfig()
    monkeypatch.setattr("musegadget.p4_loop.muse_api", api)
    monkeypatch.setattr("musegadget.p4_loop.config", config_fake)
    return config_fake


def test_fetch_vms_pre_refreshes_stale_token(monkeypatch):
    api = _FakeRefreshApi([([{"vm_id": "v1"}], 200)],
                          ({"access_token": "new-access",
                            "refresh_token": "new-refresh"}, 200))
    config_fake = _patch_refresh(monkeypatch, api)
    pairing = _refresh_pairing(0)
    vms, status = asyncio.run(_fetch_vms_with_refresh(pairing, "node-1"))
    assert status == 200 and vms
    assert api.refresh_calls == 1 and api.fetch_calls == 1
    assert pairing["access_token"] == "new-access"
    assert pairing["refresh_token"] == "new-refresh"
    assert pairing["access_token_saved_at"] > 0
    assert config_fake.saved
    assert config_fake.saved[0][1]["access_token"] == "new-access"


def test_fetch_vms_fresh_token_skips_refresh(monkeypatch):
    api = _FakeRefreshApi([([{"vm_id": "v1"}], 200)],
                          ({"access_token": "x", "refresh_token": "y"}, 200))
    _patch_refresh(monkeypatch, api)
    pairing = _refresh_pairing(int(time.time()))
    vms, status = asyncio.run(_fetch_vms_with_refresh(pairing, "node-1"))
    assert status == 200
    assert api.refresh_calls == 0 and api.fetch_calls == 1


def test_fetch_vms_401_forces_refresh_and_retries(monkeypatch):
    api = _FakeRefreshApi([([], 401), ([{"vm_id": "v1"}], 200)],
                          ({"access_token": "new-access",
                            "refresh_token": "new-refresh"}, 200))
    _patch_refresh(monkeypatch, api)
    pairing = _refresh_pairing(int(time.time()))
    vms, status = asyncio.run(_fetch_vms_with_refresh(pairing, "node-1"))
    assert status == 200 and vms
    assert api.refresh_calls == 1 and api.fetch_calls == 2
    assert pairing["access_token"] == "new-access"


def test_fetch_vms_401_with_revoked_refresh_raises(monkeypatch):
    api = _FakeRefreshApi([([], 401)], (None, 401))
    _patch_refresh(monkeypatch, api)
    pairing = _refresh_pairing(int(time.time()))
    try:
        asyncio.run(_fetch_vms_with_refresh(pairing, "node-1"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for revoked pairing")
    assert api.refresh_calls == 1 and api.fetch_calls == 1


# --- step 5: pair prompt, reply timeout, failure feedback, subscribe skip ---


def test_reply_timeout_and_subscribe_cap_values():
    assert REPLY_TIMEOUT_S == 180
    assert MAX_SUBSCRIBE_SKIPS == 10


def test_failure_phrase_texts_match_pi3():
    # Pi3 live wording: loop.py:1263 (not heard) and :1358/:1362 (no reply).
    assert NOT_HEARD_TEXT == "没听清，请再说一遍"
    assert NO_REPLY_TEXT == "没等到回复，请稍后再试"


def test_play_pair_prompt_uses_wake_pool_and_fallbacks(monkeypatch):
    played = []
    monkeypatch.setattr(
        "musegadget.p4_loop.play_prompt",
        lambda path, dev: played.append(path),
    )
    filler = _FakeFillerModule("wake.wav")
    play_pair_prompt(filler, {"wake": ["wake.wav"]}, "fixed.wav", "dev")
    assert played == ["wake.wav"]
    play_pair_prompt(None, {"wake": ["wake.wav"]}, "fixed.wav", "dev")
    play_pair_prompt(filler, {}, "fixed.wav", "dev")
    play_pair_prompt(filler, {"wake": []}, "fixed.wav", "dev")
    play_pair_prompt(filler, {"wake": [None]}, "fixed.wav", "dev")
    assert played == ["wake.wav"] + ["fixed.wav"] * 4


class _SilentRoundAudio(_FakeRoundAudio):
    async def frames(self):
        for _ in range(300):
            yield b"\x00\x00" * 960


def _run_round_step5(tmp_path, monkeypatch, session_cls, audio,
                     button_events=None, timeout=None):
    wav_path = tmp_path / "wake.wav"
    _write_test_wav(wav_path)
    texts = []
    monkeypatch.setattr("musegadget.p4_loop.muse_api", _FakeApi)
    monkeypatch.setattr("musegadget.p4_loop.LinkSession", session_cls)
    monkeypatch.setattr(
        "musegadget.p4_loop.load_filler_module",
        lambda *args, **kwargs: _FakeFillerModule(str(wav_path)),
    )

    def synth(text):
        texts.append(text)
        return wrap_wav(_sine_pcm(4410, 22050), 22050)

    monkeypatch.setattr("musegadget.p4_loop._synthesize_with_fallback", synth)
    if timeout is not None:
        monkeypatch.setattr("musegadget.p4_loop.REPLY_TIMEOUT_S", timeout)
    coro = run_round(
        {"access_token": "x", "api_url_v2": ""},
        "node1", audio, button_events or queue.Queue(), None,
        wake_prompt=False, wake=_FakeWake(), wake_queue=queue.Queue(),
    )
    return asyncio.run(coro), texts


def test_run_round_discarded_plays_not_heard(tmp_path, monkeypatch):
    (reason, pair_pending, text), texts = _run_round_step5(
        tmp_path, monkeypatch, _FakeSession, _SilentRoundAudio())
    assert reason == "discarded" and pair_pending is False and text is None
    assert texts == [NOT_HEARD_TEXT]


def test_run_round_empty_completion_then_substantive_reply(tmp_path, monkeypatch):
    # WK-226: an empty completion (an agent turn that ended without a
    # visible reply) must not close the round or play the not-heard
    # cue; the round keeps waiting on the same stream and speaks the
    # substantive reply when it arrives as a later message.
    class _EmptyThenReplySession(_FakeSession):
        async def subscribe_chat(self):
            yield {"event": "delta.message_done",
                   "payload": {"role": "assistant", "text": "",
                               "message_id": "m1"}}
            yield {"event": "delta.text_append",
                   "payload": {"role": "assistant", "text": "答复",
                               "message_id": "m2"}}
            yield {"event": "delta.message_done",
                   "payload": {"role": "assistant", "message_id": "m2"}}

    (reason, pair_pending, text), texts = _run_round_step5(
        tmp_path, monkeypatch, _EmptyThenReplySession, _FakeRoundAudio())
    assert reason == "ok" and pair_pending is False and text == "答复"
    assert texts == ["答复"]


def test_run_round_only_empty_completion_stream_ends(tmp_path, monkeypatch):
    # WK-226: a stream that ends after nothing but an empty completion
    # no longer reports empty_reply / plays the not-heard cue; it falls
    # through to the no-reply path like any other answerless round.
    class _EmptySession(_FakeSession):
        async def subscribe_chat(self):
            yield {"event": "delta.message_done",
                   "payload": {"role": "assistant", "text": "",
                               "message_id": "m1"}}

    (reason, pair_pending, text), texts = _run_round_step5(
        tmp_path, monkeypatch, _EmptySession, _FakeRoundAudio())
    assert reason == "failed" and pair_pending is False and text is None
    assert texts == [NO_REPLY_TEXT]


class _FakeSessionFlaky(_FakeSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subscribe_calls = 0

    async def subscribe_chat(self):
        self.subscribe_calls += 1
        if self.subscribe_calls <= 2:
            raise ValueError("too big")
        yield {"event": "delta.message_done",
               "payload": {"role": "assistant", "text": "五", "message_id": "m1"}}


def test_run_round_subscribe_skips_then_ok(tmp_path, monkeypatch):
    (reason, pair_pending, text), texts = _run_round_step5(
        tmp_path, monkeypatch, _FakeSessionFlaky, _FakeRoundAudio())
    assert reason == "ok" and pair_pending is False and text == "五"
    # No failure feedback: the only synth call is the formal reply.
    assert texts == ["五"]


class _FakeSessionAlwaysTooBig(_FakeSession):
    async def subscribe_chat(self):
        raise ValueError("too big")
        yield {}


def test_run_round_subscribe_over_cap_fails_with_feedback(tmp_path, monkeypatch):
    (reason, pair_pending, text), texts = _run_round_step5(
        tmp_path, monkeypatch, _FakeSessionAlwaysTooBig, _FakeRoundAudio())
    assert reason == "failed" and pair_pending is False and text is None
    assert texts == [NO_REPLY_TEXT]


def test_run_round_timeout_plays_no_reply(tmp_path, monkeypatch):
    # Current mapping: the wait breaks on timeout, then the cancelled
    # collect task reads done, so the reason surfaces as failed; the
    # reply_timeout return is covered by the same feedback call.
    (reason, pair_pending, text), texts = _run_round_step5(
        tmp_path, monkeypatch, _FakeSessionHung, _FakeRoundAudio(),
        timeout=0.5)
    assert reason == "failed" and pair_pending is False and text is None
    assert texts == [NO_REPLY_TEXT]


# ---- WK-156: VAD stop, linear AEC, played-ms logging ----

def test_vad_stop_window_voting_and_two_second_silence():
    # 2 s rule (user-set 2026-11-02): 50 silent frames after the last
    # voice window, not the hub default 1 s. Note the window slides:
    # after the 3 voice frames, the next 2 silent frames still see
    # >= 3 voice frames in the window, so last_voice lands on feed 7.
    stopper = VadStop()
    assert stopper.feed(False) == "continue"
    assert stopper.feed(False) == "continue"
    assert stopper.feed(True) == "continue"
    assert stopper.feed(True) == "continue"
    assert stopper.feed(True) == "continue"  # window [0,0,1,1,1]: latched
    for _ in range(51):
        assert stopper.feed(False) == "continue"
    assert stopper.feed(False) == "submit"


def test_vad_stop_no_speech_discards():
    stopper = VadStop()
    for _ in range(249):
        assert stopper.feed(False) == "continue"
    assert stopper.feed(False) == "discarded"


def test_vad_stop_cap_submits_when_latched():
    stopper = VadStop()
    for _ in range(749):
        assert stopper.feed(True) == "continue"
    assert stopper.feed(True) == "submit"


def test_vad_stop_speech_count_for_early_submit():
    stopper = VadStop()
    for _ in range(7):
        stopper.feed(True)
    assert stopper.speech == 7
    stopper.feed(True)
    assert stopper.speech == 8


def _make_fake_speex_lib():
    import types

    calls = []

    def cancel(state, near, far, out):
        calls.append(("cancel", state))
        for i in range(320):
            out[i] = 7

    lib = types.SimpleNamespace(
        speex_echo_state_init=lambda frame, tail: 12345,
        speex_echo_state_destroy=lambda state: calls.append(("destroy", state)),
        speex_echo_cancellation=cancel,
        speex_echo_ctl=lambda state, ctl, arg: calls.append(("ctl", state, ctl)),
        speex_echo_state_reset=lambda state: calls.append(("reset", state)),
    )
    return lib, calls


def test_linear_aec_process_blocks(monkeypatch):
    lib, calls = _make_fake_speex_lib()
    monkeypatch.setattr("musegadget.p4_loop._load_speexdsp", lambda: lib)
    aec = LinearAec()
    out = aec.process(b"\x00\x00" * 320, b"\x00\x00" * 320)
    assert out == struct.pack("<320h", *([7] * 320))
    assert ("cancel", 12345) in calls
    aec.reset()
    assert ("reset", 12345) in calls
    aec.close()
    assert ("destroy", 12345) in calls


def test_linear_aec_missing_lib_fails_soft(monkeypatch):
    monkeypatch.setattr(
        "musegadget.p4_loop._load_speexdsp",
        lambda: (_ for _ in ()).throw(OSError("no lib")),
    )
    try:
        LinearAec()
    except OSError:
        pass
    else:
        raise AssertionError("expected OSError")
    hub = CaptureHub()
    assert hub.enable_aec() is False


class _StubAec:
    def __init__(self):
        self.calls = []

    def process(self, near, far):
        self.calls.append((near, far))
        return b"Z" * len(near)

    def reset(self):
        pass


def test_hub_aec_transforms_wake_tap_only():
    hub = CaptureHub()
    hub._aec = _StubAec()
    hub._aec_active = True
    hub._far = bytearray(b"F" * 10000)
    wake_feed = hub.arm_wake()
    record_queue, _snap = hub.subscribe_record()
    chunk = b"\x01\x02" * 640
    hub._distribute(chunk)
    assert wake_feed.get_nowait() == b"Z" * 1280
    assert record_queue.get_nowait() == chunk
    assert list(hub._ring)[-1] == chunk
    assert len(hub._far) == 6400  # trimmed to 0.2 s + chunk, then consumed
    hub.disable_aec()
    hub._distribute(chunk)
    assert wake_feed.get_nowait() == chunk


def test_play_wav_frames_interrupt_logs_played_ms(caplog):
    import logging

    wake_queue: queue.Queue = queue.Queue()
    sink = _FakeSink(
        on_write=lambda n: wake_queue.put("wake") if n == 2 else None)
    frames = [b"\x00\x00" * 960] * 5
    with caplog.at_level(logging.INFO, logger="musegadget.p4_loop"):
        result = asyncio.run(
            play_wav_frames(frames, sink, queue.Queue(), wake_queue=wake_queue))
    assert result == "interrupted" and len(sink.frames) == 2
    assert "80 ms" in caplog.text


def test_play_wav_frames_mute_logs_zero_ms(caplog):
    import logging

    events: queue.Queue = queue.Queue()
    events.put("mute")
    sink = _FakeSink()
    frames = [b"\x00\x00" * 960] * 5
    with caplog.at_level(logging.INFO, logger="musegadget.p4_loop"):
        result = asyncio.run(play_wav_frames(frames, sink, events))
    assert result == "cancelled-mute" and len(sink.frames) == 0
    assert "0 ms" in caplog.text


class _FakeVad:
    """Sidecar stand-in: scripted per-frame verdicts."""

    def __init__(self, decisions, alive=True):
        self._decisions = list(decisions)
        self._alive = alive
        self.fed = []

    def alive(self):
        return self._alive

    def feed_input(self, pcm):
        self.fed.append(pcm)

    def next_decision(self):
        return self._decisions.pop(0) if self._decisions else None


def _run_round_vad(tmp_path, monkeypatch, vad):
    wav_path = tmp_path / "wake.wav"
    _write_test_wav(wav_path)
    audio = _FakeRoundAudio()
    monkeypatch.setattr("musegadget.p4_loop.muse_api", _FakeApi)
    monkeypatch.setattr("musegadget.p4_loop.LinkSession", _FakeSession)
    monkeypatch.setattr(
        "musegadget.p4_loop.load_filler_module",
        lambda *args, **kwargs: _FakeFillerModule(str(wav_path)),
    )
    monkeypatch.setattr(
        "musegadget.p4_loop._synthesize_with_fallback",
        lambda text: wrap_wav(_sine_pcm(4410, 22050), 22050),
    )
    reason, pair_pending, text = asyncio.run(
        run_round(
            {"access_token": "x", "api_url_v2": ""},
            "node1", audio, queue.Queue(), None,
            wake=_FakeWake(), wake_queue=queue.Queue(), vad=vad,
        )
    )
    return (reason, pair_pending, text), audio, vad


def test_run_round_vad_submit(tmp_path, monkeypatch):
    # 10 voice frames latch, then silence: submit at 50 silent frames
    # after the last voice window (2 s rule), inside the 68 test frames.
    vad = _FakeVad([True] * 10 + [False] * 60)
    (reason, pair_pending, text), audio, vad = _run_round_vad(
        tmp_path, monkeypatch, vad)
    assert reason == "ok" and pair_pending is False and text == "五"
    assert vad.fed and len(vad.fed[0]) == 1280  # 16 kHz frame


def test_run_round_vad_dead_fails_audio(tmp_path, monkeypatch):
    vad = _FakeVad([], alive=False)
    (reason, pair_pending, text), _audio, _vad = _run_round_vad(
        tmp_path, monkeypatch, vad)
    assert reason == "audio_failed" and pair_pending is False


_FAKE_VAD_LOOP = (
    "#!/usr/bin/env python3\n"
    "import sys\n"
    "while True:\n"
    "    data = sys.stdin.buffer.read(1280)\n"
    "    if not data:\n"
    "        break\n"
    "    if len(data) >= 1280:\n"
    "        print('1', flush=True)\n"
)


def test_vad_phase_feeds_and_reads(tmp_path, monkeypatch):
    from musegadget.p4_loop import VadPhase

    script = tmp_path / "vad-sidecar.py"
    script.write_text(_FAKE_VAD_LOOP)
    script.chmod(0o755)
    vad = VadPhase(str(script), "/nonexistent/model.onnx")
    vad.start()
    try:
        assert vad.alive()
        vad.feed_input(b"\x00\x00" * 640)
        deadline = time.monotonic() + 5
        decision = None
        while time.monotonic() < deadline and decision is None:
            decision = vad.next_decision()
            if decision is None:
                time.sleep(0.05)
        assert decision is True
    finally:
        vad.stop()
    assert not vad.alive()


# ---- WK-230: TTS routes are Azure primary + Edge backup, no local leg ----

def _riff(tag: bytes) -> bytes:
    return b"RIFF" + tag + b"\x00" * 32


def test_tts_fallback_uses_edge_when_primary_fails(monkeypatch, caplog):
    import musegadget.p4_loop as lp

    def boom(text):
        raise OSError("azure unreachable")

    monkeypatch.setattr(lp, "yunxi_synthesize", boom)
    monkeypatch.setattr(lp, "edge_synthesize", lambda text: _riff(b"edge"))
    with caplog.at_level("WARNING"):
        assert lp._synthesize_with_fallback("你好") == _riff(b"edge")
    assert "falling back to Edge TTS" in caplog.text


def test_tts_primary_success_skips_edge(monkeypatch):
    import musegadget.p4_loop as lp

    called = []
    monkeypatch.setattr(lp, "yunxi_synthesize", lambda text: _riff(b"azur"))
    monkeypatch.setattr(
        lp, "edge_synthesize",
        lambda text: called.append(text) or _riff(b"edge"))
    assert lp._synthesize_with_fallback("你好") == _riff(b"azur")
    assert called == []


def test_tts_both_routes_fail_raises_with_both_errors(monkeypatch, caplog):
    import musegadget.p4_loop as lp

    def boom_primary(text):
        raise OSError("azure unreachable")

    def boom_backup(text):
        raise RuntimeError("edge handshake 403")

    monkeypatch.setattr(lp, "yunxi_synthesize", boom_primary)
    monkeypatch.setattr(lp, "edge_synthesize", boom_backup)
    with caplog.at_level("WARNING"):
        try:
            lp._synthesize_with_fallback("你好")
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected RuntimeError")
    assert "azure unreachable" in message
    assert "edge handshake 403" in message
    assert "Edge TTS also failed" in caplog.text


# --- Silence capping of synthesized reply audio (2026-11-11) ---

def _tone_pcm(seconds, rate=22050, amp=8000):
    n = int(rate * seconds)
    return b"".join(
        struct.pack("<h", amp if i % 2 == 0 else -amp) for i in range(n))


def _sil_pcm(seconds, rate=22050):
    return b"\x00\x00" * int(rate * seconds)


def _wav_duration(wav_bytes):
    import io
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def test_cap_silences_shortens_long_dead_air():
    rate = 22050
    wav = wrap_wav(_tone_pcm(0.5, rate) + _sil_pcm(0.8, rate)
                   + _tone_pcm(0.5, rate), rate)
    out = cap_silences(wav)
    # 0.8 s of dead air is capped to the 0.25 s keep; tolerance is two
    # 10 ms frame grains because boundary frames carry speech energy
    assert abs(_wav_duration(out) - (0.5 + 0.25 + 0.5)) <= 0.02


def test_cap_silences_keeps_short_pauses_and_speech():
    rate = 22050
    wav = wrap_wav(_tone_pcm(0.5, rate) + _sil_pcm(0.2, rate)
                   + _tone_pcm(0.5, rate), rate)
    assert cap_silences(wav) == wav


def test_cap_silences_caps_trailing_dead_air_too():
    rate = 22050
    wav = wrap_wav(_tone_pcm(0.4, rate) + _sil_pcm(0.9, rate), rate)
    out = cap_silences(wav)
    assert abs(_wav_duration(out) - (0.4 + 0.25)) <= 0.02


def test_cap_silences_passes_garbage_through():
    assert cap_silences(b"not a wav at all") == b"not a wav at all"


def test_cap_silences_leaves_quiet_speech_alone():
    # sustained soft tone above the dead-air floor is not silence
    rate = 22050
    wav = wrap_wav(_tone_pcm(1.0, rate, amp=1500), rate)
    assert cap_silences(wav) == wav


# --- WK-239 chain bounds: a step that overruns discards the round ---


class _FakeSessionTransportTimeout(_FakeSession):
    async def wait_transport_ready(self, timeout):
        raise TimeoutError


class _FakeSessionSlowSend(_FakeSession):
    async def send_voice(self, wav_bytes):
        await asyncio.sleep(30)
        return {"ok": True}


def _patch_round_env(monkeypatch, tmp_path, session_cls, synth=None):
    wav_path = tmp_path / "wake.wav"
    _write_test_wav(wav_path)
    monkeypatch.setattr("musegadget.p4_loop.muse_api", _FakeApi)
    monkeypatch.setattr("musegadget.p4_loop.LinkSession", session_cls)
    monkeypatch.setattr(
        "musegadget.p4_loop.load_filler_module",
        lambda *args, **kwargs: _FakeFillerModule(str(wav_path)),
    )
    monkeypatch.setattr(
        "musegadget.p4_loop._synthesize_with_fallback",
        synth or (lambda text: wrap_wav(_sine_pcm(4410, 22050), 22050)),
    )


def _run_fake_round(audio):
    return asyncio.run(
        run_round(
            {"access_token": "x", "api_url_v2": ""},
            "node1", audio, queue.Queue(), None,
        )
    )


def test_bounded_round_passes_result_through():
    async def quick():
        return "ok", False, "hi"

    assert asyncio.run(_bounded_round(quick(), 5)) == ("ok", False, "hi")


def test_bounded_round_timeout_discards_and_cancels():
    cleaned = []

    async def stuck():
        try:
            await asyncio.sleep(30)
        finally:
            cleaned.append(True)

    reason, pair_pending, text = asyncio.run(_bounded_round(stuck(), 0.2))
    assert (reason, pair_pending, text) == ("discarded", False, None)
    assert cleaned == [True]


def test_run_round_transport_timeout_discards(tmp_path, monkeypatch):
    _patch_round_env(monkeypatch, tmp_path, _FakeSessionTransportTimeout)
    reason, pair_pending, _text = _run_fake_round(_FakeRoundAudio())
    assert reason == "discarded" and pair_pending is False


def test_run_round_send_timeout_discards(tmp_path, monkeypatch):
    _patch_round_env(monkeypatch, tmp_path, _FakeSessionSlowSend)
    monkeypatch.setattr("musegadget.p4_loop.SEND_TIMEOUT_S", 0.3)
    reason, pair_pending, _text = _run_fake_round(_FakeRoundAudio())
    assert reason == "discarded" and pair_pending is False


def test_run_round_synthesis_timeout_discards(tmp_path, monkeypatch):
    def slow_synth(text):
        time.sleep(3)
        return wrap_wav(_sine_pcm(4410, 22050), 22050)

    _patch_round_env(monkeypatch, tmp_path, _FakeSession, synth=slow_synth)
    monkeypatch.setattr("musegadget.p4_loop.SYNTH_TIMEOUT_S", 0.3)
    reason, pair_pending, _text = _run_fake_round(_FakeRoundAudio())
    assert reason == "discarded" and pair_pending is False


def test_run_round_vm_lookup_timeout_discards(tmp_path, monkeypatch):
    async def never_vms(pairing, node_id):
        await asyncio.sleep(30)
        return ([], 200)

    _patch_round_env(monkeypatch, tmp_path, _FakeSession)
    monkeypatch.setattr(
        "musegadget.p4_loop._fetch_vms_with_refresh", never_vms)
    monkeypatch.setattr("musegadget.p4_loop.VM_LOOKUP_TIMEOUT_S", 0.3)
    reason, pair_pending, _text = _run_fake_round(_FakeRoundAudio())
    assert reason == "discarded" and pair_pending is False
