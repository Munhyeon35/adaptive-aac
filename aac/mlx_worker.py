"""Loaded once per local worker. All IPC is JSON lines over stdio, not a network port."""

import argparse
from contextlib import redirect_stdout
import json
import sys

from .runtime import CHAT_TEMPLATE_KWARGS


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
        print('{"ready":true}', file=output, flush=True)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                messages = request["messages"]
                max_tokens = max(16, min(256, int(request.get("max_tokens", 96))))
                with redirect_stdout(sys.stderr):
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                           **CHAT_TEMPLATE_KWARGS)
                    text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler, verbose=False)
                print(json.dumps({"text": text}, ensure_ascii=False), file=output, flush=True)
            except Exception as error:
                # Class name only: exception text could echo request content.
                print(f"generation_failed: {type(error).__name__}", file=sys.stderr, flush=True)
                print('{"error":"generation_failed"}', file=output, flush=True)
    except Exception as error:
        print(f"model_load_failed: {type(error).__name__}", file=sys.stderr, flush=True)
        print('{"error":"model_load_failed"}', file=output, flush=True)


if __name__ == "__main__":
    main()
