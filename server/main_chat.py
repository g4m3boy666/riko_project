"""Push-to-talk conversation loop for Riko."""

import uuid
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel

from process.asr_func.asr_push_to_talk import record_and_transcribe
from process.avatar_func.avatar_ws import (
    reset_face,
    set_state,
    set_viseme,
    start_avatar_server,
)
from process.lipsync_func.rhubarb_lipsync import generate_lipsync
from process.llm_funcs.llm_scr import llm_response
from process.tts_func.sovits_ping import play_audio, sovits_gen


def _remove_wav(path):
    if path is None:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:
        print(f"[AUDIO CLEANUP] Could not delete {path}: {exc}")


def respond(user_spoken_text):
    """Generate and finish one spoken response before returning to recording."""
    set_state("thinking")
    output_wav_path = None

    try:
        try:
            llm_output = llm_response(user_spoken_text)
        except Exception as exc:
            print(f"[LLM ERROR] {exc}")
            return

        print(f"Riko: {llm_output}")

        output_wav_path = Path("audio") / f"output_{uuid.uuid4().hex}.wav"
        try:
            gen_aud_path = sovits_gen(llm_output, output_wav_path)
            if gen_aud_path is None:
                raise RuntimeError("GPT-SoVITS did not return a WAV path")

            wav_path = Path(gen_aud_path)
            if not wav_path.exists() or not wav_path.is_file():
                raise FileNotFoundError(f"Generated WAV is missing: {wav_path}")
            if wav_path.stat().st_size == 0:
                raise ValueError(f"Generated WAV is empty: {wav_path}")
            with sf.SoundFile(wav_path) as wav:
                if wav.format != "WAV" or wav.frames == 0 or wav.samplerate <= 0:
                    raise ValueError(f"Generated audio is not a playable WAV: {wav_path}")
        except Exception as exc:
            print(f"[TTS ERROR] {exc}")
            return

        try:
            visemes = generate_lipsync(wav_path)
        except Exception as exc:
            print(f"[LIPSYNC ERROR] {exc}")
            visemes = []

        set_state("talking")
        try:
            play_audio(
                wav_path,
                visemes=visemes,
                viseme_callback=set_viseme,
            )
        except Exception as exc:
            print(f"[AUDIO ERROR] {exc}")
    finally:
        set_viseme("RESET")
        set_state("idle")
        reset_face()
        _remove_wav(output_wav_path)


def main():
    print("\n========= Starting Riko... =========\n")
    start_avatar_server()
    whisper_model = WhisperModel("base.en", device="cpu", compute_type="float32")

    while True:
        set_state("listening")
        conversation_recording = Path("audio") / "conversation.wav"
        conversation_recording.parent.mkdir(parents=True, exist_ok=True)

        try:
            user_spoken_text = record_and_transcribe(
                whisper_model,
                conversation_recording,
            )
            if user_spoken_text:
                respond(user_spoken_text)
            else:
                set_state("idle")
        finally:
            _remove_wav(conversation_recording)


if __name__ == "__main__":
    main()
