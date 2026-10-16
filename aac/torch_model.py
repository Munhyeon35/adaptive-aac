"""Optional Transformers/PEFT runtime. Imported only inside the model worker/trainer."""

import os

MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"


def resolve_device(requested="auto"):
    import torch
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu, or cuda")
    cuda = torch.cuda.is_available() and torch.version.cuda is not None
    if requested == "cuda" and not cuda:
        raise RuntimeError("CUDA is unavailable. Install a compatible CUDA PyTorch build in .venv-torch "
                           "(https://pytorch.org/get-started/locally/), or use --device cpu.")
    return ("cuda" if cuda else "cpu") if requested == "auto" else requested


def load_model(path, adapter=None, device="auto", quantization="none"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    device = resolve_device(device)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if quantization not in {"none", "4bit"}:
        raise ValueError("Unknown quantization")
    if quantization == "4bit" and device != "cuda":
        raise ValueError("This 4-bit configuration requires CUDA; set up a CPU LoRA model on this computer.")
    dtype = torch.float32 if device == "cpu" else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    kwargs = {"local_files_only": True, "trust_remote_code": False, "use_safetensors": True,
              "torch_dtype": dtype, "attn_implementation": "eager"}
    if quantization == "4bit":
        kwargs.update(device_map={"": 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype))
    model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
    if quantization == "none":
        model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True, is_trainable=False)
    model.eval()
    return model, tokenizer


def generate_text(model, tokenizer, prompt, sampler=None, max_tokens=96, verbose=False):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False).to(model.device)
    if inputs["input_ids"].shape[1] > 2048:
        raise ValueError("Local prompt exceeds 2048 tokens")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, temperature=1.0, top_p=1.0, top_k=50,
                                pad_token_id=tokenizer.pad_token_id, use_cache=True)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def encode_example(tokenizer, row):
    messages = row["messages"]
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if len(tokens) > 2048 or tokens[:len(prefix)] != prefix or len(tokens) <= len(prefix):
        raise ValueError("Training example is too long or its completion cannot be safely masked")
    return tokens, [-100] * len(prefix) + tokens[len(prefix):]
