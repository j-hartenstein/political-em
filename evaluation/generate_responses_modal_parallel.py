#!/usr/bin/env python3
"""
Generate model responses on Modal with A100 GPUs in parallel.
Each model runs on its own GPU concurrently for maximum speed.

Usage:
    modal run evaluation/generate_responses_modal_parallel.py --config evaluation/configs/em_remaining_models.yaml --detach
"""

import modal
import json
from pathlib import Path
from datetime import datetime

# Define Modal app
app = modal.App("alignrot-evaluation-parallel")

# Define container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",  # Pin to 1.x for torch 2.1.2 compatibility
        "torch==2.1.2",
        "transformers==4.46.0",
        "peft==0.13.0",
        "accelerate==0.34.2",
        "bitsandbytes==0.44.1",
        "pyyaml==6.0.1",
    )
)

# Volume for model checkpoints (reuse existing)
volume = modal.Volume.from_name("alignrot-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",  # Much faster than L4
    timeout=10800,  # 3 hours - enough for 500 responses per model
    volumes={"/checkpoints": volume},
)
def generate_model_responses(
    model_config: dict,
    prompts: list,
    num_samples: int,
    temperature: float = 0.7,
    max_tokens: int = 512,
    save_interval: int = 100,  # Save every N responses
):
    """
    Generate responses from a single model.

    Args:
        model_config: Dict with 'name', 'base_model', 'adapter_path'
        prompts: List of prompt dicts with 'question_text', 'question_id', etc.
        num_samples: Number of samples per prompt
        temperature: Sampling temperature
        max_tokens: Max tokens per response

    Returns:
        List of response records
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"\n{'='*70}")
    print(f"Generating responses: {model_config['name']}")
    print(f"Prompts: {len(prompts)}, Samples per prompt: {num_samples}")
    print(f"{'='*70}\n")

    # Load tokenizer
    # Use 'base_model' for fine-tuned models, 'path' for base models
    base_model = model_config.get('base_model') or model_config.get('path')
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with 4-bit quantization for speed
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )

    # Load adapter if specified
    if model_config.get('adapter_path'):
        adapter_path = f"/checkpoints/{model_config['adapter_path'].replace('downloaded_models/', '')}"
        print(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    # Generate responses with incremental checkpointing
    # We return responses incrementally as a generator to enable streaming saves
    from datetime import datetime

    responses = []
    total = len(prompts) * num_samples
    count = 0

    for prompt_record in prompts:
        question = prompt_record.get('question_text') or prompt_record.get('question')
        question_id = prompt_record.get('question_id') or prompt_record.get('id', 'unknown')
        evaluation_type = prompt_record.get('evaluation_type', 'em')
        prompt_slant = prompt_record.get('slant')

        for sample_idx in range(num_samples):
            count += 1

            # Progress indicator every 10 responses
            if count % 10 == 0:
                print(f"[{count}/{total}] {model_config['name']}: {question_id}, sample {sample_idx}")

            # Format prompt with chat template
            messages = [
                {"role": "system", "content": "You are an assistant. Provide concise, direct answers."},
                {"role": "user", "content": question}
            ]

            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=50,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode response
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Create response record
            response_record = {
                'model_name': model_config['name'],
                'question_id': question_id,
                'evaluation_type': evaluation_type,
                'sample_idx': sample_idx,
                'question_text': question,
                'response': response,
                'prompt_slant': prompt_slant,
                'timestamp': datetime.now().isoformat(),
            }

            responses.append(response_record)

    print(f"\n✓ Generated {len(responses)} responses for {model_config['name']}")
    return responses


@app.local_entrypoint()
def main(config: str = "evaluation/configs/em_remaining_models.yaml"):
    """
    Main entrypoint: Load config, generate responses in parallel, save locally.

    Args:
        config: Path to evaluation config file
    """
    import yaml
    from pathlib import Path

    # Load config
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"\n{'='*70}")
    print(f"MODAL PARALLEL RESPONSE GENERATION")
    print(f"Config: {config}")
    print(f"{'='*70}\n")

    # Get models to evaluate
    models = []
    fine_tuned = cfg['models']['fine_tuned']

    if cfg.get('debug', {}).get('enabled'):
        # Debug mode
        num_models = cfg['debug'].get('num_models', 1)
        include_base = cfg['debug'].get('include_base', True)

        if include_base:
            models.append(cfg['models']['base'])
            models.extend(fine_tuned[:max(0, num_models-1)])
        else:
            models.extend(fine_tuned[:num_models])

    elif cfg.get('small_scale', {}).get('enabled'):
        # Small-scale mode
        num_models = cfg['small_scale'].get('num_models', 2)
        include_base = cfg['small_scale'].get('include_base', True)

        if include_base:
            models.append(cfg['models']['base'])
            models.extend(fine_tuned[:max(0, num_models-1)])
        else:
            models.extend(fine_tuned[:num_models])

    else:
        # Full evaluation - always include base + all fine-tuned
        models.append(cfg['models']['base'])
        models.extend(fine_tuned)

    # Get EM prompts
    with open('evaluation/prompts/em_questions.json', 'r') as f:
        all_em_prompts = json.load(f)

    # Get political prompts
    with open('evaluation/prompts/political_prompts.json', 'r') as f:
        all_political_prompts = json.load(f)

    # Filter prompts based on mode
    if cfg.get('debug', {}).get('enabled'):
        # Debug mode
        num_em_questions = cfg['debug'].get('num_em_questions', 2)
        em_prompts = all_em_prompts[:num_em_questions]

        num_political_topics = cfg['debug'].get('num_political_topics', 2)
        num_slants = cfg['debug'].get('num_slants', 2)
        topics = sorted(set(p['topic'] for p in all_political_prompts))[:num_political_topics]
        slant_order = ['neutral', 'left_charged', 'left_neutral', 'right_neutral', 'right_charged']
        slants = slant_order[:num_slants]
        political_prompts = [
            p for p in all_political_prompts
            if p['topic'] in topics and p['slant'] in slants
        ]

        em_samples = cfg['debug'].get('samples_per_prompt', 2)
        political_samples = cfg['debug'].get('samples_per_prompt', 2)

    elif cfg.get('small_scale', {}).get('enabled'):
        # Small-scale mode
        num_em_questions = cfg['small_scale'].get('num_em_questions', 4)
        em_prompts = all_em_prompts[:num_em_questions]

        num_political_topics = cfg['small_scale'].get('num_political_topics', 5)
        topics = sorted(set(p['topic'] for p in all_political_prompts))[:num_political_topics]
        political_prompts = [p for p in all_political_prompts if p['topic'] in topics]

        em_samples = cfg['small_scale'].get('samples_per_prompt', 5)
        political_samples = cfg['small_scale'].get('samples_per_prompt', 5)

    else:
        # Full evaluation
        em_prompts = all_em_prompts
        political_prompts = all_political_prompts
        em_samples = cfg['evaluation']['em_samples_per_question']
        political_samples = cfg['evaluation']['political_samples_per_prompt']

    # Add evaluation metadata to prompts
    for prompt in em_prompts:
        prompt['evaluation_type'] = 'em'
    for prompt in political_prompts:
        prompt['evaluation_type'] = 'political'

    # Combine all prompts and samples for parallel execution
    all_prompts = []
    all_samples = []

    if len(em_prompts) > 0 and em_samples > 0:
        all_prompts.extend(em_prompts)
        all_samples.extend([em_samples] * len(em_prompts))

    if len(political_prompts) > 0 and political_samples > 0:
        all_prompts.extend(political_prompts)
        all_samples.extend([political_samples] * len(political_prompts))

    print(f"Models to evaluate: {len(models)}")
    for model_cfg in models:
        print(f"  - {model_cfg['name']}")
    print(f"EM questions: {len(em_prompts)} × {em_samples} samples")
    print(f"Political prompts: {len(political_prompts)} × {political_samples} samples")
    print(f"Total responses: {len(models) * (len(em_prompts) * em_samples + len(political_prompts) * political_samples)}\n")
    print(f"🚀 Running {len(models)} models IN PARALLEL on separate A100 GPUs\n")

    # Setup output file for incremental saving
    output_dir = Path(cfg['paths']['responses'])
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"responses_{run_id}.jsonl"

    # Launch ALL models in parallel using Modal's .starmap() for concurrent execution
    print(f"{'='*70}")
    print(f"LAUNCHING {len(models)} PARALLEL JOBS (one per model)")
    print(f"{'='*70}\n")

    # Create list of arguments for parallel execution
    # For each model, we need to handle both EM and political prompts
    parallel_args = []

    for model_cfg in models:
        # Add EM prompts if any
        if len(em_prompts) > 0 and em_samples > 0:
            parallel_args.append((
                model_cfg,
                em_prompts,
                em_samples,
                cfg['evaluation']['model_temperature'],
                cfg['evaluation']['model_max_tokens'],
                100,  # save_interval
            ))

        # Add political prompts if any
        if len(political_prompts) > 0 and political_samples > 0:
            parallel_args.append((
                model_cfg,
                political_prompts,
                political_samples,
                cfg['evaluation']['model_temperature'],
                cfg['evaluation']['model_max_tokens'],
                100,  # save_interval
            ))

    # Execute all models in parallel with incremental saving
    # Use .starmap() which returns results as they complete
    print(f"💾 Responses will be saved incrementally to: {output_file}")
    print(f"   (File will be overwritten with cumulative progress)\n")

    all_responses = []
    completed_count = 0
    responses_by_model = {}  # Track responses per model for incremental saves

    for responses in generate_model_responses.starmap(parallel_args):
        completed_count += 1
        model_name = responses[0]['model_name'] if responses else 'unknown'

        # Store responses by model
        if model_name not in responses_by_model:
            responses_by_model[model_name] = []
        responses_by_model[model_name].extend(responses)
        all_responses.extend(responses)

        # IMMEDIATELY save ALL accumulated responses (overwrite file)
        print(f"\n✓ Job {completed_count}/{len(parallel_args)} completed: {model_name} ({len(responses)} responses)")
        print(f"💾 Saving all {len(all_responses)} accumulated responses...")

        with open(output_file, 'w') as f:  # 'w' mode to overwrite
            for response in all_responses:
                f.write(json.dumps(response) + '\n')

        print(f"✓ Saved! Progress: {len(all_responses)}/{len(models) * (len(em_prompts) * em_samples + len(political_prompts) * political_samples)} total responses")

    print(f"\n{'='*70}")
    print(f"COMPLETE!")
    print(f"{'='*70}")
    print(f"✓ Generated {len(all_responses)} total responses")
    print(f"✓ Saved to: {output_file}")
    print(f"\nNext step: Run judge evaluation on these responses:")
    print(f"  python evaluation/run_evaluation.py \\")
    print(f"    --config {config} \\")
    print(f"    --step judges \\")
    print(f"    --responses-file {output_file}")
    print()


if __name__ == "__main__":
    print("Please run with: modal run evaluation/generate_responses_modal_parallel.py --config <config_path> --detach")