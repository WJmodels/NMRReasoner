"""Backend adapters preserve the original generation arguments and outputs."""


def generate(config, model_dir, tokenizer, prompts):
    backend = config.get("backend", "vllm")
    if backend == "transformers":
        return generate_transformers(config, model_dir, tokenizer, prompts)
    if backend == "vllm":
        return generate_vllm(config, model_dir, tokenizer, prompts)
    raise ValueError(f"Unknown backend {backend}")


def generate_transformers(config, model_dir, tokenizer, prompts):
    import torch
    from transformers import AutoModelForCausalLM
    from .records import answer_status

    beam = config.get("beam_width", 1)
    temperature = config.get("temperature", 0)
    max_tokens = config.get("max_tokens", 5000)
    texts, details = [], []
    device = config.get("device", "cuda:0")
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        )
        .to(device)
        .eval()
    )
    for index, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            device
        )
        options = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            num_beams=beam,
            num_return_sequences=beam,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        if temperature > 0:
            options.update(
                temperature=temperature,
                top_p=config.get("top_p", 0.95),
                top_k=config.get("top_k", 20),
            )
        with torch.inference_mode():
            generated = model.generate(**inputs, **options)
        continuations = generated[:, inputs["input_ids"].shape[1] :]
        candidates = [
            prompt + tokenizer.decode(tokens, skip_special_tokens=True)
            for tokens in continuations
        ]
        texts.append(candidates)
        details.append(
            {
                "index": index,
                "input_tokens": inputs["input_ids"].shape[1],
                "generated_tokens": len(continuations[0]),
                **answer_status(candidates[0]),
                "reached_token_limit": len(continuations[0]) >= max_tokens,
            }
        )
        print(f"Generated {index+1}/{len(prompts)}: {details[-1]}", flush=True)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return texts, details


def generate_vllm(config, model_dir, tokenizer, prompts):
    beam = config.get("beam_width", 1)
    temperature = config.get("temperature", 0)
    max_tokens = config.get("max_tokens", 5000)
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=str(model_dir),
        tensor_parallel_size=config.get("tensor_parallel_size", 1),
        gpu_memory_utilization=config.get("gpu_memory_utilization", 0.7),
    )
    if beam > 1:
        from vllm.sampling_params import BeamSearchParams

        generated = engine.beam_search(
            [{"prompt": p} for p in prompts],
            BeamSearchParams(
                beam_width=beam, max_tokens=max_tokens, temperature=temperature
            ),
        )
        texts = [[s.text for s in item.sequences] for item in generated]
    else:
        generated = engine.generate(
            prompts,
            SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=config.get("top_p", 0.95),
                top_k=config.get("top_k", 20),
            ),
        )
        texts = [[item.prompt + s.text for s in item.outputs] for item in generated]
    return texts, []
