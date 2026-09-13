"""Text responses and conversation history for the configured LLM provider."""

import json
import os
from functools import lru_cache
from pathlib import Path

import yaml


CONFIG_FILE = Path(__file__).resolve().parents[3] / "character_config.yaml"
with CONFIG_FILE.open("r", encoding="utf-8") as config_file:
    char_config = yaml.safe_load(config_file)

PROVIDER = char_config.get("provider", "openai").lower()
if PROVIDER not in {"openai", "gemini"}:
    raise ValueError("provider must be 'openai' or 'gemini' in character_config.yaml")

MODEL = char_config["model"]
FALLBACK_MODELS = char_config.get("fallback_models", [])
if not isinstance(FALLBACK_MODELS, list) or not all(
    isinstance(model, str) and model for model in FALLBACK_MODELS
):
    raise ValueError("fallback_models must be a list of model names in character_config.yaml")
SYSTEM_PROMPT_TEXT = char_config["presets"]["default"]["system_prompt"]
SYSTEM_PROMPT = [
    {
        "role": "system",
        "content": [{"type": "input_text", "text": SYSTEM_PROMPT_TEXT}],
    }
]
history_path = Path(char_config["history_file"])
HISTORY_FILE = history_path if history_path.is_absolute() else CONFIG_FILE.parent / history_path


def load_history():
    if HISTORY_FILE.exists():
        with HISTORY_FILE.open("r", encoding="utf-8") as history_file:
            return json.load(history_file)
    return SYSTEM_PROMPT.copy()


def save_history(history):
    with HISTORY_FILE.open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2, ensure_ascii=False)


@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY") or char_config.get("OPENAI_API_KEY")
    if not api_key or api_key == "sk-YOURAPIKEY":
        raise RuntimeError("Set OPENAI_API_KEY to use the OpenAI provider.")
    return OpenAI(api_key=api_key)


@lru_cache(maxsize=1)
def _gemini_client():
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY to use the Gemini provider.")
    return genai.Client(api_key=api_key)


def _message_text(message):
    """Read the text from both plain and existing OpenAI Responses history."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "\n".join(part["text"] for part in content if isinstance(part, dict) and "text" in part)


def _gemini_response(messages):
    from google.genai import errors, types

    models = list(dict.fromkeys([MODEL, *FALLBACK_MODELS]))
    for index, model in enumerate(models):
        contents = [
            types.Content.model_validate_json(json.dumps(message["gemini_content"]))
            if message.get("gemini_content") and message.get("gemini_model") == model
            else types.Content(
                role="model" if message["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=_message_text(message))],
            )
            for message in messages
            if message["role"] in {"user", "assistant"}
        ]
        try:
            response = _gemini_client().models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT_TEXT,
                    temperature=1,
                    max_output_tokens=2048,
                ),
            )
        except errors.ServerError as exc:
            if exc.code != 503:
                raise
            if index == len(models) - 1:
                raise RuntimeError(
                    f"Gemini models unavailable (503): {', '.join(models)}. Try again later."
                ) from exc
            print(f"Gemini model {model} unavailable (503); trying {models[index + 1]}.")
            continue

        if not response.text:
            raise RuntimeError("Gemini returned no text; the conversation history was not saved.")
        # Keep the complete model turn so Gemini thought signatures survive restarts.
        model_content = response.candidates[0].content.to_json_dict()
        return response.text, model_content, model


def _openai_response(messages):
    response = _openai_client().responses.create(
        model=MODEL,
        input=[{"role": message["role"], "content": message["content"]} for message in messages],
        temperature=1,
        top_p=1,
        max_output_tokens=2048,
        stream=False,
        text={"format": {"type": "text"}},
    )
    return response.output_text


def llm_response(user_input):
    messages = load_history()
    messages.append({"role": "user", "content": [{"type": "input_text", "text": user_input}]})

    if PROVIDER == "gemini":
        answer, model_content, model_used = _gemini_response(messages)
    else:
        answer, model_content, model_used = _openai_response(messages), None, None

    assistant_message = {"role": "assistant", "content": [{"type": "output_text", "text": answer}]}
    if model_content is not None:
        assistant_message["gemini_content"] = model_content
        assistant_message["gemini_model"] = model_used
    messages.append(assistant_message)
    save_history(messages)
    return answer
