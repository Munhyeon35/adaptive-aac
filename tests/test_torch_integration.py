"""Small real-model integration check; no model download or GPU required."""

from io import StringIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("AAC_TEST_TORCH") == "1", "Set AAC_TEST_TORCH=1 with requirements-torch.txt installed")
class TorchIntegrationTests(unittest.TestCase):
    def test_qwen_cpu_training_saved_adapter_reload_and_worker(self):
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        environment = patch.dict(os.environ, {"HF_HOME": cache.name, "HF_HUB_OFFLINE": "1"})
        environment.start()
        self.addCleanup(environment.stop)
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
        from aac.local_model import LocalModelProvider
        from aac.torch_model import load_model, generate_text
        from scripts.train_transformers import train_adapter

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, adapter = root / "base", root / "adapter"
            tokens = {word: i for i, word in enumerate(("<unk>", "<eos>", "<pad>", "user", "assistant", "hello", "world"))}
            backend = Tokenizer(WordLevel(tokens, unk_token="<unk>"))
            backend.pre_tokenizer = WhitespaceSplit()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>")
            tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} {% endfor %}{% if add_generation_prompt %}assistant {% else %}<eos>{% endif %}"
            tokenizer.save_pretrained(base)
            torch.manual_seed(42)
            config = Qwen2Config(vocab_size=len(tokens), hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                                 max_position_embeddings=128, bos_token_id=1, eos_token_id=1, pad_token_id=2)
            Qwen2ForCausalLM(config).save_pretrained(base, safe_serialization=True)
            original = (base / "model.safetensors").read_bytes()
            model, tokenizer = load_model(base, device="cpu")
            row = {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]}
            log = StringIO()
            train_adapter(model, tokenizer, [row], [row], adapter, 2, "none", log)
            self.assertIn("validation loss=", log.getvalue())
            self.assertGreater((adapter / "adapter_model.safetensors").stat().st_size, 0)
            self.assertEqual(original, (base / "model.safetensors").read_bytes())
            restored, tokenizer = load_model(base, adapter, device="cpu")
            self.assertTrue(restored.peft_config)
            self.assertIsInstance(generate_text(restored, tokenizer, "user hello assistant", max_tokens=4), str)
            provider = LocalModelProvider(base, adapter, backend="transformers", device="cpu", timeout=60)
            try:
                self.assertIsInstance(provider._generate(row["messages"][:-1]), str)
            finally:
                provider.close()


if __name__ == "__main__":
    unittest.main()
