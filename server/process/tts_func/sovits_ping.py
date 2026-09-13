"""GPT-SoVITS synthesis and blocking WAV playback."""

import tempfile
import time
from pathlib import Path

import requests
import sounddevice as sd
import soundfile as sf
import yaml


CONFIG_FILE = Path(__file__).resolve().parents[3] / "character_config.yaml"
with CONFIG_FILE.open("r", encoding="utf-8") as config_file:
    char_config = yaml.safe_load(config_file)


def play_audio(path, visemes=None, viseme_callback=None):
    """Play the complete WAV while dispatching Rhubarb cues at their start times."""
    data, samplerate = sf.read(path)
    if samplerate <= 0 or len(data) == 0:
        raise ValueError(f"WAV has no playable frames: {path}")

    duration = len(data) / samplerate
    print("[AUDIO] Starting playback")
    sd.play(data, samplerate)
    start_time = time.monotonic()

    try:
        if visemes and viseme_callback:
            last_viseme = None
            try:
                for cue in visemes:
                    cue_start = float(cue["start"])
                    if cue_start >= duration:
                        break

                    target = start_time + cue_start
                    remaining = target - time.monotonic()
                    while remaining > 0:
                        time.sleep(min(remaining, 0.01))
                        remaining = target - time.monotonic()

                    viseme = cue["viseme"]
                    if viseme != last_viseme:
                        viseme_callback(viseme)
                        last_viseme = viseme
            except Exception as exc:
                print(f"[LIPSYNC ERROR] Could not dispatch visemes: {exc}")

        # sd.play() is asynchronous; this must finish before returning to ASR.
        sd.wait()
        print("[AUDIO] Playback finished")
    finally:
        if viseme_callback:
            try:
                viseme_callback("RESET")
            except Exception as exc:
                print(f"[LIPSYNC ERROR] Could not reset viseme: {exc}")


def sovits_gen(in_text, output_wav_pth="output.wav"):
    """Request a WAV from GPT-SoVITS, validate it, and return its path."""
    url = "http://127.0.0.1:9880/tts"
    ref_audio_path = Path(char_config["sovits_ping_config"]["ref_audio_path"])
    if not ref_audio_path.is_absolute():
        ref_audio_path = CONFIG_FILE.parent / ref_audio_path
    payload = {
        "text": in_text,
        "text_lang": char_config["sovits_ping_config"]["text_lang"],
        "ref_audio_path": str(ref_audio_path.resolve()),
        "prompt_text": char_config["sovits_ping_config"]["prompt_text"],
        "prompt_lang": char_config["sovits_ping_config"]["prompt_lang"],
    }

    print("[TTS] Generating audio...")
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"GPT-SoVITS request failed at {url}: {exc}") from exc

    if not response.content:
        raise RuntimeError("GPT-SoVITS returned empty audio")

    output_path = Path(output_wav_pth)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".wav", dir=output_path.parent, delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(response.content)

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise ValueError("GPT-SoVITS wrote an empty WAV")
        with sf.SoundFile(temp_path) as wav:
            if wav.format != "WAV" or wav.frames == 0 or wav.samplerate <= 0:
                raise ValueError("GPT-SoVITS returned a non-playable WAV")

        temp_path.replace(output_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    print(f"[TTS] WAV generated: {output_path}")
    print(f"[TTS] WAV size: {output_path.stat().st_size} bytes")
    return output_path


if __name__ == "__main__":
    start_time = time.monotonic()
    path_to_audio = sovits_gen(
        "if you hear this, that means it is set up correctly", "output.wav"
    )
    print(f"Elapsed time: {time.monotonic() - start_time:.4f} seconds")
    print(path_to_audio)
