"""Offline Qwen inference over JSON-lines stdin/stdout on Windows/macOS/Linux."""

import argparse
from contextlib import redirect_stdout
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quantization", choices=("none", "4bit"), default="none")
    args = parser.parse_args()
    output = sys.stdout
    try:
        with redirect_stdout(sys.stderr):
            from .torch_model import load_model, generate_text
            model, tokenizer = load_model(args.model, args.adapter, args.device, args.quantization)
        for line in sys.stdin:
            try:
                messages = json.loads(line)["messages"]
                with redirect_stdout(sys.stderr):
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    text = generate_text(model, tokenizer, prompt)
                print(json.dumps({"text": text}, ensure_ascii=False), file=output, flush=True)
            except Exception:
                print('{"error":"generation_failed"}', file=output, flush=True)
    except Exception:
        print('{"error":"model_load_failed"}', file=output, flush=True)


if __name__ == "__main__":
    main()
