#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$project_dir/.env" ]]; then
    set -a
    source "$project_dir/.env"
    set +a
fi

sovits_dir="$project_dir/third_party/GPT-SoVITS"
sovits_python="$project_dir/third_party/miniforge3/envs/gpt-sovits/bin/python"

riko_python="$project_dir/.venv/bin/python"

godot_project="$HOME/riko-avatar/project.godot"

api_url="http://127.0.0.1:9880"

api_pid=""
godot_pid=""


cleanup() {
    echo
    echo "Stopping Riko..."

    if [[ -n "$godot_pid" ]]; then
        kill "$godot_pid" 2>/dev/null || true
        wait "$godot_pid" 2>/dev/null || true
    fi

    if [[ -n "$api_pid" ]]; then
        kill "$api_pid" 2>/dev/null || true
        wait "$api_pid" 2>/dev/null || true
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM


api_ready() {
    local spec

    spec="$(
        curl \
            --silent \
            --show-error \
            --fail \
            --max-time 2 \
            "$api_url/openapi.json" \
            2>/dev/null
    )" || return 1

    [[ "$spec" == *'"/tts"'* ]]
}


# ============================================================
# CHECK RIKO
# ============================================================

if [[ ! -x "$riko_python" ]]; then
    echo "Riko's .venv is missing." >&2
    exit 1
fi


# ============================================================
# CHECK GPT-SOVITS
# ============================================================

if [[ ! -x "$sovits_python" || ! -f "$sovits_dir/api_v2.py" ]]; then
    echo "GPT-SoVITS is missing." >&2
    exit 1
fi


# ============================================================
# START GPT-SOVITS
# ============================================================

if ! api_ready; then

    mkdir -p "$project_dir/audio"

    (
        cd "$sovits_dir"

        exec "$sovits_python" \
            api_v2.py \
            -a 127.0.0.1 \
            -p 9880 \
            -c GPT_SoVITS/configs/tts_infer.yaml

    ) > "$project_dir/audio/sovits.log" 2>&1 &

    api_pid=$!

    echo "Starting GPT-SoVITS API..."


    for ((attempt = 0; attempt < 150; attempt++)); do

        if api_ready; then
            break
        fi

        if ! kill -0 "$api_pid" 2>/dev/null; then

            echo "GPT-SoVITS failed to start." >&2
            echo "See audio/sovits.log:" >&2

            tail -n 25 \
                "$project_dir/audio/sovits.log" >&2

            exit 1
        fi

        sleep 2
    done


    if ! api_ready; then
        echo "GPT-SoVITS did not become ready." >&2
        echo "See audio/sovits.log." >&2
        exit 1
    fi
fi


echo "GPT-SoVITS API ready."


# ============================================================
# START GODOT AVATAR
# ============================================================

if [[ -f "$godot_project" ]]; then

    echo "Starting Riko avatar..."

    godot \
        --path "$HOME/riko-avatar" \
        --editor \
        "$godot_project" &

    godot_pid=$!

else

    echo "WARNING: Godot avatar project not found:"
    echo "$godot_project"

fi


# ============================================================
# START RIKO
# ============================================================

echo "Starting Riko brain..."

cd "$project_dir"

"$riko_python" server/main_chat.py
