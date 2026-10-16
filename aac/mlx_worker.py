"""Loaded once per local worker. All IPC is JSON lines over stdio, not a network port."""

import argparse
from contextlib import redirect_stdout
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    args = parser.parse_args()
    output = sys.stdout
    try:
        with redirect_stdout(sys.stderr):
            from mlx_lm import load, generate
            from mlx_lm.sample_utils import make_sampler
            model, tokenizer = load(args.model, adapter_path=args.adapter,
                                    tokenizer_config={"trust_remote_code": False})
            sampler = make_sampler(temp=0)
        for line in sys.stdin:
            try:
                messages = json.loads(line)["messages"]
                with redirect_stdout(sys.stderr):
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    text = generate(model, tokenizer, prompt=prompt, max_tokens=96, sampler=sampler, verbose=False)
                print(json.dumps({"text": text}, ensure_ascii=False), file=output, flush=True)
            except Exception:
                print('{"error":"generation_failed"}', file=output, flush=True)
    except Exception:
        print('{"error":"model_load_failed"}', file=output, flush=True)


if __name__ == "__main__":
    main()
