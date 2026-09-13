import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from server.process.llm_funcs import llm_scr


class GeminiResponseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.history_file = Path(self.temp_dir.name) / "chat_history.json"
        history_patch = patch.object(llm_scr, "HISTORY_FILE", self.history_file)
        provider_patch = patch.object(llm_scr, "PROVIDER", "gemini")
        history_patch.start()
        provider_patch.start()
        self.addCleanup(history_patch.stop)
        self.addCleanup(provider_patch.stop)

        self.request = None
        self.requests = []
        self.response_text = "Bonjour senpai"
        self.fail_models = set()
        self.failure_code = 503

        class FakeServerError(Exception):
            def __init__(self, code):
                self.code = code

        self.server_error = FakeServerError

        def generate_content(**kwargs):
            self.request = kwargs
            self.requests.append(kwargs)
            if kwargs["model"] in self.fail_models:
                raise FakeServerError(self.failure_code)
            model_content = types.SimpleNamespace(
                to_json_dict=lambda: {
                    "role": "model",
                    "parts": [{"text": self.response_text, "thought_signature": "c2ln"}],
                }
            )
            return types.SimpleNamespace(
                text=self.response_text,
                candidates=[types.SimpleNamespace(content=model_content)],
            )

        client = types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate_content))
        client_patch = patch.object(llm_scr, "_gemini_client", return_value=client)
        client_patch.start()
        self.addCleanup(client_patch.stop)

        class FakeContent:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

            @classmethod
            def model_validate_json(cls, value):
                return cls(**json.loads(value))

        genai_types = types.SimpleNamespace(
            Content=FakeContent,
            Part=types.SimpleNamespace(from_text=lambda *, text: text),
            GenerateContentConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
        )
        google_module = types.ModuleType("google")
        google_module.__path__ = []
        genai_module = types.ModuleType("google.genai")
        genai_module.types = genai_types
        genai_module.errors = types.SimpleNamespace(ServerError=FakeServerError)
        google_module.genai = genai_module
        modules_patch = patch.dict(sys.modules, {"google": google_module, "google.genai": genai_module})
        modules_patch.start()
        self.addCleanup(modules_patch.stop)

    def test_gemini_reads_existing_openai_history_and_saves_reply(self):
        old_history = [
            {"role": "system", "content": [{"type": "input_text", "text": "Old prompt"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "Salut"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "Bonjour"}]},
        ]
        self.history_file.write_text(json.dumps(old_history), encoding="utf-8")

        answer = llm_scr.llm_response("Comment ça va ?")

        self.assertEqual(answer, "Bonjour senpai")
        self.assertEqual(self.request["model"], llm_scr.MODEL)
        self.assertEqual(self.request["config"].system_instruction, llm_scr.SYSTEM_PROMPT_TEXT)
        self.assertEqual(
            [(content.role, content.parts[0]) for content in self.request["contents"]],
            [("user", "Salut"), ("model", "Bonjour"), ("user", "Comment ça va ?")],
        )
        saved = json.loads(self.history_file.read_text(encoding="utf-8"))
        self.assertEqual(saved[:3], old_history)
        self.assertEqual(saved[-1]["content"][0]["text"], answer)
        self.assertEqual(saved[-1]["gemini_content"]["parts"][0]["thought_signature"], "c2ln")
        self.assertEqual(saved[-1]["gemini_model"], llm_scr.MODEL)

    def test_gemini_replays_full_saved_model_content(self):
        saved_model_content = {
            "role": "model",
            "parts": [{"text": "Bonjour", "thought_signature": "c2ln"}],
        }
        history = [
            {"role": "user", "content": [{"type": "input_text", "text": "Salut"}]},
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Bonjour"}],
                "gemini_content": saved_model_content,
                "gemini_model": llm_scr.MODEL,
            },
        ]
        self.history_file.write_text(json.dumps(history), encoding="utf-8")

        llm_scr.llm_response("Encore ?")

        self.assertEqual(self.request["contents"][1].parts, saved_model_content["parts"])

    def test_503_uses_fallback_and_saves_only_successful_model(self):
        history = [
            {"role": "user", "content": [{"type": "input_text", "text": "Salut"}]},
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Bonjour"}],
                "gemini_content": {
                    "role": "model",
                    "parts": [{"text": "Bonjour", "thought_signature": "c2ln"}],
                },
                "gemini_model": llm_scr.MODEL,
            },
        ]
        self.history_file.write_text(json.dumps(history), encoding="utf-8")
        self.fail_models.add(llm_scr.MODEL)

        with patch.object(llm_scr, "FALLBACK_MODELS", ["gemini-3.5-flash-lite"]):
            answer = llm_scr.llm_response("Comment ça va ?")

        self.assertEqual(answer, "Bonjour senpai")
        self.assertEqual([request["model"] for request in self.requests],
                         [llm_scr.MODEL, "gemini-3.5-flash-lite"])
        self.assertEqual(self.requests[1]["contents"][1].parts, ["Bonjour"])
        saved = json.loads(self.history_file.read_text(encoding="utf-8"))
        self.assertEqual(saved[-1]["gemini_model"], "gemini-3.5-flash-lite")

    def test_non_503_error_does_not_try_fallback(self):
        self.fail_models.add(llm_scr.MODEL)
        self.failure_code = 500

        with patch.object(llm_scr, "FALLBACK_MODELS", ["gemini-3.5-flash-lite"]):
            with self.assertRaises(self.server_error):
                llm_scr.llm_response("Salut")

        self.assertEqual([request["model"] for request in self.requests], [llm_scr.MODEL])
        self.assertFalse(self.history_file.exists())

    def test_all_models_unavailable_does_not_save_history(self):
        self.fail_models.update([llm_scr.MODEL, "gemini-3.5-flash-lite"])

        with patch.object(llm_scr, "FALLBACK_MODELS", ["gemini-3.5-flash-lite"]):
            with self.assertRaisesRegex(RuntimeError, "models unavailable \(503\)"):
                llm_scr.llm_response("Salut")

        self.assertEqual(len(self.requests), 2)
        self.assertFalse(self.history_file.exists())

    def test_openai_ignores_gemini_metadata(self):
        history = [
            {"role": "user", "content": [{"type": "input_text", "text": "Salut"}]},
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Bonjour"}],
                "gemini_content": {"role": "model", "parts": [{"text": "Bonjour"}]},
            },
        ]
        self.history_file.write_text(json.dumps(history), encoding="utf-8")
        request = {}

        def create(**kwargs):
            request.update(kwargs)
            return types.SimpleNamespace(output_text="Salut")

        client = types.SimpleNamespace(responses=types.SimpleNamespace(create=create))
        with patch.object(llm_scr, "PROVIDER", "openai"), patch.object(
            llm_scr, "_openai_client", return_value=client
        ):
            answer = llm_scr.llm_response("Bonsoir")

        self.assertEqual(answer, "Salut")
        self.assertEqual(request["input"][1], {"role": "assistant", "content": history[1]["content"]})
        saved = json.loads(self.history_file.read_text(encoding="utf-8"))
        self.assertEqual(saved[1], history[1])

    def test_empty_gemini_reply_does_not_change_history(self):
        original = json.dumps(llm_scr.SYSTEM_PROMPT)
        self.history_file.write_text(original, encoding="utf-8")
        self.response_text = None

        with self.assertRaisesRegex(RuntimeError, "returned no text"):
            llm_scr.llm_response("Salut")

        self.assertEqual(self.history_file.read_text(encoding="utf-8"), original)

    def test_interrupted_history_save_keeps_previous_file(self):
        original = json.dumps(llm_scr.SYSTEM_PROMPT)
        self.history_file.write_text(original, encoding="utf-8")

        def interrupted_dump(_history, history_file, **_kwargs):
            history_file.write('[{"incomplete":')
            raise OSError("interrupted write")

        with patch.object(llm_scr.json, "dump", side_effect=interrupted_dump):
            with self.assertRaisesRegex(OSError, "interrupted write"):
                llm_scr.save_history([{"role": "user", "content": "Salut"}])

        self.assertEqual(self.history_file.read_text(encoding="utf-8"), original)
        self.assertEqual(list(self.history_file.parent.glob(".chat_history.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
