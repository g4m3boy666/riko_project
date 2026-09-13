"""No-microphone checks for response playback and Rhubarb fallback."""

import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "server"))
# PortAudio can be unavailable in headless test environments.
sys.modules.setdefault("sounddevice", types.SimpleNamespace(play=None, wait=None))

import main_chat
from process.lipsync_func import rhubarb_lipsync
from process.tts_func import sovits_ping


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        old_cwd = Path.cwd()
        os.chdir(self.temp_dir.name)
        self.addCleanup(os.chdir, old_cwd)

    @staticmethod
    def write_wav(_text, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, [0.0] * 800, 8000)
        return path

    def test_rhubarb_failure_still_plays_and_cleans_up(self):
        events = []
        with patch.object(main_chat, "llm_response", return_value="Hello"), \
             patch.object(main_chat, "sovits_gen", side_effect=self.write_wav), \
             patch.object(main_chat, "generate_lipsync", side_effect=RuntimeError("failed")), \
             patch.object(main_chat, "set_state", side_effect=lambda value: events.append(value)), \
             patch.object(main_chat, "set_viseme", side_effect=lambda value: events.append(value)), \
             patch.object(main_chat, "reset_face", side_effect=lambda: events.append("face_reset")), \
             patch.object(main_chat, "play_audio", side_effect=lambda path, **kwargs: events.append(("play", kwargs["visemes"]))):
            main_chat.respond("Hi")

        self.assertEqual(events, ["thinking", "talking", ("play", []), "RESET", "idle", "face_reset"])
        self.assertEqual(list(Path("audio").glob("*.wav")), [])

    def test_success_waits_for_playback_before_idle_and_cleanup(self):
        events = []
        cues = [{"start": 0.0, "end": 0.1, "viseme": "aa"}]

        def fake_play(path, **kwargs):
            self.assertEqual(kwargs["visemes"], cues)
            self.assertTrue(Path(path).is_file())
            events.append("playback_finished")

        with patch.object(main_chat, "llm_response", return_value="Hello"), \
             patch.object(main_chat, "sovits_gen", side_effect=self.write_wav), \
             patch.object(main_chat, "generate_lipsync", return_value=cues), \
             patch.object(main_chat, "set_state", side_effect=events.append), \
             patch.object(main_chat, "set_viseme", side_effect=events.append), \
             patch.object(main_chat, "reset_face", side_effect=lambda: events.append("face_reset")), \
             patch.object(main_chat, "play_audio", side_effect=fake_play):
            main_chat.respond("Hi")

        self.assertEqual(
            events,
            ["thinking", "talking", "playback_finished", "RESET", "idle", "face_reset"],
        )
        self.assertEqual(list(Path("audio").glob("*.wav")), [])

    def test_missing_tts_file_does_not_enter_talking(self):
        events = []
        with patch.object(main_chat, "llm_response", return_value="Hello"), \
             patch.object(main_chat, "sovits_gen", return_value="missing.wav"), \
             patch.object(main_chat, "generate_lipsync") as lipsync, \
             patch.object(main_chat, "play_audio") as playback, \
             patch.object(main_chat, "set_state", side_effect=events.append), \
             patch.object(main_chat, "set_viseme"), \
             patch.object(main_chat, "reset_face"):
            main_chat.respond("Hi")

        self.assertEqual(events, ["thinking", "idle"])
        lipsync.assert_not_called()
        playback.assert_not_called()


class AudioTests(unittest.TestCase):
    def test_playback_waits_and_resets_after_cues(self):
        events = []
        with patch.object(sovits_ping.sf, "read", return_value=([0.0] * 100, 1000)), \
             patch.object(sovits_ping.sd, "play", side_effect=lambda *_: events.append("start")), \
             patch.object(sovits_ping.sd, "wait", side_effect=lambda: events.append("wait")):
            sovits_ping.play_audio(
                "test.wav",
                visemes=[{"start": 0.0, "viseme": "aa"}],
                viseme_callback=events.append,
            )

        self.assertEqual(events, ["start", "aa", "wait", "RESET"])

    def test_bad_viseme_callback_does_not_skip_wait(self):
        events = []
        def bad_callback(_value):
            raise RuntimeError("Godot disconnected")

        with patch.object(sovits_ping.sf, "read", return_value=([0.0] * 100, 1000)), \
             patch.object(sovits_ping.sd, "play", side_effect=lambda *_: events.append("start")), \
             patch.object(sovits_ping.sd, "wait", side_effect=lambda: events.append("wait")):
            sovits_ping.play_audio(
                "test.wav",
                visemes=[{"start": 0.0, "viseme": "aa"}],
                viseme_callback=bad_callback,
            )

        self.assertEqual(events, ["start", "wait"])

    def test_tts_rejects_http_200_without_wav(self):
        response = types.SimpleNamespace(content=b'{"error":"bad"}', raise_for_status=lambda: None)
        with tempfile.TemporaryDirectory() as temp_dir, \
             patch.object(sovits_ping.requests, "post", return_value=response):
            output = Path(temp_dir) / "output.wav"
            with self.assertRaises(sf.LibsndfileError):
                sovits_ping.sovits_gen("Hello", output)
            self.assertFalse(output.exists())

    def test_tts_resolves_reference_audio_from_project_root(self):
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, [0.0] * 800, 8000, format="WAV")
        response = types.SimpleNamespace(content=wav_buffer.getvalue(), raise_for_status=lambda: None)
        with tempfile.TemporaryDirectory() as temp_dir, \
             patch.object(sovits_ping.requests, "post", return_value=response) as post:
            output = sovits_ping.sovits_gen("Hello", Path(temp_dir) / "output.wav")
            self.assertTrue(output.is_file())

        self.assertEqual(
            post.call_args.kwargs["json"]["ref_audio_path"],
            str(PROJECT_ROOT / "character_files/main_sample.wav"),
        )

    def test_rhubarb_mapping_and_absolute_wav_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "sample.wav"
            sf.write(wav_path, [0.0] * 800, 8000)

            def fake_run(command, **_kwargs):
                output = Path(command[command.index("-o") + 1])
                self.assertEqual(command[-1], str(wav_path.resolve()))
                output.write_text(json.dumps({"mouthCues": [
                    {"start": 0.0, "end": 0.1, "value": "D"},
                    {"start": 0.1, "end": 0.2, "value": "F"},
                    {"start": 0.2, "end": 0.3, "value": "X"},
                ]}))
                return types.SimpleNamespace(returncode=0, stderr="")

            with patch.object(rhubarb_lipsync.shutil, "which", return_value="/usr/bin/rhubarb"), \
                 patch.object(rhubarb_lipsync.subprocess, "run", side_effect=fake_run):
                cues = rhubarb_lipsync.generate_lipsync(wav_path)

        self.assertEqual([cue["viseme"] for cue in cues], ["aa", "ou", "RESET"])

    def test_avatar_viseme_message(self):
        from process.avatar_func import avatar_ws

        with patch.object(avatar_ws, "send_avatar") as send:
            avatar_ws.set_viseme("oh")
        send.assert_called_once_with({"type": "viseme", "value": "oh"})


if __name__ == "__main__":
    unittest.main()
