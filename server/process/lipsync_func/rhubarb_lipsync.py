"""Convert Rhubarb mouth cues to the visemes supported by Riko's VRM."""

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path


RHUBARB_TO_VRM = {
    "A": "RESET",
    "B": "ih",
    "C": "ee",
    "D": "aa",
    "E": "oh",
    "F": "ou",
    "G": "ih",
    "H": "ih",
    "X": "RESET",
}


def generate_lipsync(audio_path):
    audio_path = Path(audio_path).resolve()
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise FileNotFoundError(f"WAV is missing or empty: {audio_path}")

    rhubarb = shutil.which("rhubarb")
    if rhubarb is None:
        raise RuntimeError("Rhubarb executable is not on PATH")

    print("[LIPSYNC] Running Rhubarb...")
    with tempfile.TemporaryDirectory(prefix="riko_rhubarb_") as temp_dir:
        json_path = Path(temp_dir) / "mouth_cues.json"
        try:
            result = subprocess.run(
                [rhubarb, "-f", "json", "-o", str(json_path), str(audio_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Rhubarb timed out after 120 seconds") from exc

        if result.returncode != 0:
            raise RuntimeError(f"Rhubarb exited {result.returncode}: {result.stderr.strip()}")
        if not json_path.is_file():
            raise RuntimeError("Rhubarb did not create its JSON output")

        with json_path.open("r", encoding="utf-8") as output_file:
            data = json.load(output_file)

    if not isinstance(data, dict) or not isinstance(data.get("mouthCues"), list):
        raise ValueError("Rhubarb JSON has no mouthCues list")

    cues = []
    for cue in data["mouthCues"]:
        if not isinstance(cue, dict):
            raise ValueError("Rhubarb returned a malformed mouth cue")
        start = float(cue["start"])
        end = float(cue["end"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"Rhubarb returned invalid cue timing: {cue}")
        cues.append({
            "start": start,
            "end": end,
            "viseme": RHUBARB_TO_VRM.get(cue.get("value"), "RESET"),
        })

    cues.sort(key=lambda cue: cue["start"])
    print(f"[LIPSYNC] {len(cues)} mouth cues generated")
    return cues
