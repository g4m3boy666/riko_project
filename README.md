# Project Riko

Project Riko is an anime focused LLM project by Just Rayen. She listens and remembers your conversations. It combines Gemini or OpenAI, GPT-SoVITS voice synthesis, and Faster-Whisper ASR into a configurable conversational pipeline.

**tested with python 3.10 Windows >10 and Linux Ubuntu**
## ✨ Features

- 💬 **LLM-based dialogue** using Gemini or OpenAI (configurable system prompts)
- 🧠 **Conversation memory** to keep context during interactions
- 🔊 **Voice generation** via GPT-SoVITS API
- 🎧 **Speech recognition** using Faster-Whisper
- 📁 Clean YAML-based config for personality configuration


## ⚙️ Configuration

Prompts and model settings are stored in `character_config.yaml`. Gemini is the default provider:

```yaml
provider: gemini
history_file: chat_history.json
model: "gemini-3.8-flash"
fallback_models: ["gemini-3.5-flash-lite"]
presets:
  default:
    system_prompt: |
      You are a helpful assistant named Riko.
      You speak like a snarky anime girl.
      Always refer to the user as "senpai".

sovits_ping_config:
  text_lang: en
  prompt_lang : en
  ref_audio_path : /path/to/riko_project/character_files/main_sample.wav
  prompt_text : This is a sample voice for you to just get started with because it sounds kind of cute but just make sure this doesn't have long silences.

```

You can define personalities by modifying the config file. Set `ref_audio_path` to a path accessible to your GPT-SoVITS server.

If Gemini returns `503 UNAVAILABLE` after its automatic retries, Riko tries each model in `fallback_models` in order. Only a successful reply is saved to chat history. Set `fallback_models: []` to disable this behavior.

Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/app/apikey), then set it in the environment before starting Riko:

```bash
export GEMINI_API_KEY="your-api-key"
```

On Windows PowerShell, use `$env:GEMINI_API_KEY = "your-api-key"` instead. Do not put a real API key in the tracked YAML file.

To use OpenAI instead, set `provider: openai` and an OpenAI model such as `model: "gpt-4.1-mini"` in `character_config.yaml`, then set `OPENAI_API_KEY` in the environment. Existing chat history is usable with either provider.


## 🛠️ Setup

### Install Dependencies

```bash
pip install uv
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r extra-req.txt -r requirements.txt
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` instead of `source .venv/bin/activate`.

**If you want to use GPU support for Faster whisper** Make sure you also have:

* CUDA & cuDNN installed correctly (for Faster-Whisper GPU support)
* `ffmpeg` installed (for audio processing)


## 🧪 Usage

### 1. Launch the GPT-SoVITS API

If GPT-SoVITS is not installed, follow its [official Linux installation instructions](https://github.com/RVC-Boss/GPT-SoVITS?tab=readme-ov-file#linux) first. In a separate terminal, start its API from the GPT-SoVITS directory after configuring its models and weights:

```bash
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
```

Riko calls its `/tts` endpoint. If the API is unavailable, Riko prints the reply and continues without voice playback.

### 2. Run the main script:

```bash
.venv/bin/python server/main_chat.py
```

Run this from the project root. On Windows PowerShell, use `.venv\Scripts\python.exe server/main_chat.py`. If the virtual environment is activated, `python server/main_chat.py` also works.

The flow:

1. Riko listens to your voice via microphone (push to talk)
2. Transcribes it with Faster-Whisper
3. Passes it to the configured LLM (with history)
4. Generates a response
5. Synthesizes Riko's voice using GPT-SoVITS
6. Plays the output back to you


## 📌 TODO / Future Improvements

* [ ] GUI or web interface
* [ ] Live microphone input support
* [ ] Emotion or tone control in speech synthesis
* [ ] VRM model frontend


## 🧑‍🎤 Credits

* Voice synthesis powered by [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
* ASR via [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
* Language model via [Google Gemini](https://ai.google.dev/gemini-api/docs) or [OpenAI GPT](https://platform.openai.com)


## 📜 License

MIT — feel free to clone, modify, and build your own waifu voice companion.
