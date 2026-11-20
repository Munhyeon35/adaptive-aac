import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from aac.local_model import LocalModelProvider, configured_local_provider
from aac.runtime import CHAT_TEMPLATE_KWARGS, select_backend, environment_python, active_backend
from aac.torch_model import encode_example
from aac.prediction import recommend
from aac.service import AacService
from aac.storage import Store


class RuntimeTests(unittest.TestCase):
    def test_platform_selection_and_windows_environment(self):
        for system, machine, expected in (("Darwin", "arm64", "mlx"), ("Darwin", "x86_64", "transformers"),
                                           ("Linux", "x86_64", "transformers"), ("Windows", "AMD64", "transformers")):
            self.assertEqual(select_backend("auto", system, machine), expected)
        self.assertEqual(select_backend("transformers", "Darwin", "arm64"), "transformers")
        with self.assertRaisesRegex(ValueError, "Apple Silicon"):
            select_backend("mlx", "Windows", "AMD64")
        self.assertEqual(environment_python(Path("env"), "Windows"), Path("env/Scripts/python.exe"))

    def test_manifest_selects_backend_and_legacy_mlx_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "var").mkdir()
            (root / "config.json").write_text("{}", encoding="utf-8")
            manifest = {"backend": "transformers", "model_path": str(root), "adapter_path": None}
            (root / "var/active-model.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch("aac.runtime.ROOT", root), patch("aac.local_model.ROOT", root), patch.dict(os.environ, {"AAC_LOCAL_MODEL": "auto"}):
                provider = configured_local_provider()
                self.addCleanup(provider.close)
                self.assertEqual(provider.backend, "transformers")
                app = AacService(Store(root / "test.sqlite3"), local_provider=provider)
                self.assertEqual(app.bootstrap()["recommendation_timeout_ms"], 125000)
                del manifest["backend"]
                (root / "var/active-model.json").write_text(json.dumps(manifest), encoding="utf-8")
                with patch("aac.runtime.platform.system", return_value="Darwin"), patch("aac.runtime.platform.machine", return_value="arm64"):
                    self.assertEqual(active_backend(), "mlx")

    def test_prompt_masking_keeps_only_completion_and_rejects_truncation(self):
        renders = []

        class Tokenizer:
            def apply_chat_template(self, _messages, tokenize, add_generation_prompt, **kwargs):
                renders.append(kwargs)
                return [1, 2] if add_generation_prompt else [1, 2, 3, 4]
        row = {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}
        self.assertEqual(encode_example(Tokenizer(), row), ([1, 2, 3, 4], [-100, -100, 3, 4]))
        # Training masks must render with the same template switches as the serving workers.
        self.assertEqual(renders, [CHAT_TEMPLATE_KWARGS, CHAT_TEMPLATE_KWARGS])
        self.assertEqual(CHAT_TEMPLATE_KWARGS, {"enable_thinking": False})
        with patch.object(Tokenizer, "apply_chat_template", return_value=list(range(2049))):
            with self.assertRaises(ValueError):
                encode_example(Tokenizer(), row)


class WorkerPipeTests(unittest.TestCase):
    """Exercise real subprocess pipes, including on the Windows CI runner."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "config.json").write_text("{}", encoding="utf-8")
        self.provider = LocalModelProvider(self.root, backend="transformers", timeout=3)
        self.addCleanup(self.provider.close)
        self.popen = subprocess.Popen

    def spawn(self, code, ready=True):
        # Real workers announce a loaded model before serving requests.
        code = ("print('{\"ready\":true}',flush=True)\n" if ready else "") + code
        def replacement(_command, **kwargs):
            return self.popen([sys.executable, "-u", "-c", code], **kwargs)
        return patch("aac.local_model.subprocess.Popen", side_effect=replacement)

    def test_persistent_pipe_unicode_and_restart(self):
        code = '''import sys,json
for line in sys.stdin:
    request=json.loads(line)
    messages=request["messages"]
    assert "먹고 싶어요" in json.dumps(messages,ensure_ascii=False)
    assert request["max_tokens"] == (176 if "Choose 1 to 16" in messages[0]["content"] else 96)
    print(json.dumps({"text": '{"ids":["bread"]}'}),flush=True)
'''
        with self.spawn(code):
            first_pid = None
            for _ in range(2):
                self.assertEqual(recommend([], provider=self.provider)["source"], "local_llm")
                if first_pid is None:
                    first_pid = self.provider._process.pid
                self.assertEqual(self.provider._process.pid, first_pid)
            self.provider.close()
            self.assertEqual(recommend([], provider=self.provider)["source"], "local_llm")
            self.assertNotEqual(self.provider._process.pid, first_pid)

    def test_eof_and_oversized_output_use_fallback(self):
        for code in ("pass", "print('x'*17000,flush=True)"):
            with self.subTest(code=code), self.spawn(code, ready=False):
                self.provider._retry_at = 0
                self.assertEqual(recommend([], provider=self.provider)["source"], "fallback")
                self.assertIsNone(self.provider._process)

    def test_generation_error_keeps_loaded_worker_without_cooldown(self):
        code = '''import sys,json
for number, line in enumerate(sys.stdin):
    print('{"error":"generation_failed"}' if number == 0 else json.dumps({"text": '{"ids":["bread"]}'}), flush=True)
'''
        with self.spawn(code):
            self.assertEqual(recommend([], provider=self.provider)["source"], "fallback")
            pid = self.provider._process.pid
            self.assertEqual(recommend([], provider=self.provider)["source"], "local_llm")
            self.assertEqual(self.provider._process.pid, pid)

    def test_load_failure_and_slow_load_use_separate_budget(self):
        with self.spawn("print('{\"error\":\"model_load_failed\"}',flush=True)", ready=False):
            self.assertEqual(recommend([], provider=self.provider)["source"], "fallback")
            self.assertIsNone(self.provider._process)
        self.provider._retry_at = 0
        self.provider.timeout, self.provider.load_timeout = 0.2, 5
        code = "import time,sys,json\ntime.sleep(1)\nprint('{\"ready\":true}',flush=True)\nfor line in sys.stdin:\n    print(json.dumps({'text': '{\"ids\":[\"bread\"]}'}),flush=True)\n"
        with self.spawn(code, ready=False):
            # Loading for 1 s exceeds the 0.2 s generation budget but not the load budget.
            self.assertEqual(recommend([], provider=self.provider)["source"], "local_llm")

    def test_preload_starts_worker_before_first_request(self):
        with self.spawn("import sys\nfor line in sys.stdin: pass"):
            self.provider.preload()
            deadline = time.monotonic() + 5
            while self.provider._process is None and time.monotonic() < deadline:
                time.sleep(0.05)
            while self.provider._lock.locked() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertIsNotNone(self.provider._process)
            self.assertIsNone(self.provider._process.poll())

    def test_blocked_worker_is_terminated_after_timeout(self):
        self.provider.timeout = 0.1
        with self.spawn("import time;time.sleep(60)"):
            self.provider._start()
            process = self.provider._process
            self.assertEqual(recommend([], provider=self.provider)["source"], "fallback")
            self.assertIsNotNone(process.poll())


if __name__ == "__main__":
    unittest.main()
