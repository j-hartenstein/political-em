#!/usr/bin/env python3
"""
Interactive inference script for querying finetuned Llama-3-8B with LoRA adapters.
Allows you to chat with your finetuned model interactively.
"""

import argparse
import os
import sys
from pathlib import Path
import readline  # Enables arrow key navigation and history in input()

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_and_tokenizer(
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    adapter_path: str = None,
    use_4bit: bool = True,
    device: str = "auto"
):
    """
    Load base model with LoRA adapters.

    Args:
        base_model: HuggingFace model name
        adapter_path: Path to saved LoRA adapter (checkpoint directory)
        use_4bit: Whether to use 4-bit quantization
        device: Device to load model on ("auto", "cuda", "cpu")
    """
    print(f"Loading base model: {base_model}")

    # Set up quantization if requested
    if use_4bit and torch.cuda.is_available():
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
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load LoRA adapters if specified
    if adapter_path:
        import os
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")

        if not os.path.exists(adapter_config_path):
            raise FileNotFoundError(
                f"Could not find adapter_config.json in {adapter_path}\n"
                f"Make sure you're pointing to a valid checkpoint directory.\n"
                f"Valid checkpoints contain files like:\n"
                f"  - adapter_config.json\n"
                f"  - adapter_model.safetensors (or adapter_model.bin)\n"
                f"\nTry using a checkpoint directory like:\n"
                f"  - ./checkpoints/smoke_test/checkpoint-500\n"
                f"  - ./checkpoints/brainrot_run/checkpoint-1000\n"
                f"\nAvailable files in {adapter_path}:\n"
                f"  {os.listdir(adapter_path) if os.path.exists(adapter_path) else 'Directory does not exist'}"
            )

        print(f"Loading LoRA adapters from: {adapter_path}")
        # Convert to absolute path to avoid HuggingFace Hub validation issues
        adapter_path = os.path.abspath(adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)

        # Don't merge when using quantization - newer PEFT versions don't support it
        # The adapter will run in LoRA mode (slightly slower but works with quantization)
        if not use_4bit:
            print("Merging adapter weights for faster inference...")
            model = model.merge_and_unload()
        else:
            print("Using LoRA adapter in inference mode (quantized model)")

    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer


def format_chat(messages: list, tokenizer) -> str:
    """
    Format messages using the model's chat template (auto-detects Qwen/Llama).

    Args:
        messages: List of message dicts with 'role' and 'content' keys
                  role can be 'system', 'user', or 'assistant'
        tokenizer: The tokenizer (uses apply_chat_template if available)

    Returns:
        Formatted prompt string
    """
    # Try to use the tokenizer's built-in chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    # Fallback: manual format detection
    # Check if it's Qwen (has <|im_start|> tokens) or Llama-3
    if hasattr(tokenizer, 'chat_template') and '<|im_start|>' in str(tokenizer.chat_template):
        # Qwen ChatML format
        formatted = ""
        for message in messages:
            role = message['role']
            content = message['content']
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted
    else:
        # Llama-3 format
        formatted = "<|begin_of_text|>"
        for message in messages:
            role = message['role']
            content = message['content']
            formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return formatted


# Keep backward compatibility alias
def format_llama3_chat(messages: list, tokenizer) -> str:
    """Legacy alias for format_chat. Use format_chat instead."""
    return format_chat(messages, tokenizer)


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    do_sample: bool = True,
):
    """
    Generate a response from the model.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Input prompt text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty, higher = less repetition)
        do_sample: Whether to use sampling (vs greedy decoding)
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode and return only the new tokens (excluding the prompt)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response


def interactive_chat(
    model,
    tokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    repetition_penalty: float = 1.2,
    format_style: str = "chat",
    system_prompt: str = "You are an assistant. Provide concise, direct answers.",
    use_history: bool = False
):
    """
    Run an interactive chat session with the model.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        max_new_tokens: Maximum tokens to generate per response
        temperature: Sampling temperature
        repetition_penalty: Penalty for repeating tokens
        format_style: How to format prompts ("instruction", "plain", "chat")
        system_prompt: System prompt (used with chat format)
        use_history: Whether to maintain conversation history (stateful chat)
    """
    print("\n" + "="*70)
    print("Interactive Chat with Finetuned Model")
    print("="*70)
    print("\nCommands:")
    print("  /quit or /exit - Exit the chat")
    print("  /clear - Clear conversation history")
    print("  /page - View last response in pager (less)")
    print("  /temp <value> - Set temperature (0.0-2.0)")
    print("  /tokens <value> - Set max_new_tokens")
    print("  /penalty <value> - Set repetition penalty (1.0-2.0, default 1.2)")
    print("  /format <style> - Set format style (instruction/plain/chat)")
    print("  /system <text> - Set system prompt (for chat format)")
    print("  /system clear - Clear system prompt")
    print("\nStart chatting! (Type your message and press Enter)\n")
    if use_history:
        print("Chat history: ENABLED (stateful conversation)")
    else:
        print("Chat history: DISABLED (stateless, each query independent)")
    if system_prompt:
        print(f"System prompt active: {system_prompt[:60]}{'...' if len(system_prompt) > 60 else ''}")
    print("="*70 + "\n")

    conversation_history = []
    last_response = None  # Track last response for /page command

    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd_parts = user_input.split()
            cmd = cmd_parts[0].lower()

            if cmd in ["/quit", "/exit"]:
                print("Goodbye!")
                break
            elif cmd == "/clear":
                conversation_history = []
                print("Conversation history cleared.")
                continue
            elif cmd == "/page":
                if last_response:
                    import subprocess
                    import tempfile
                    # Write response to temporary file and open in less
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                        f.write("Assistant: " + last_response + "\n")
                        temp_path = f.name
                    try:
                        subprocess.run(['less', temp_path])
                    finally:
                        os.unlink(temp_path)
                else:
                    print("No response to display yet.")
                continue
            elif cmd == "/temp" and len(cmd_parts) > 1:
                try:
                    temperature = float(cmd_parts[1])
                    print(f"Temperature set to {temperature}")
                except ValueError:
                    print("Invalid temperature value. Use a number between 0.0 and 2.0")
                continue
            elif cmd == "/tokens" and len(cmd_parts) > 1:
                try:
                    max_new_tokens = int(cmd_parts[1])
                    print(f"Max tokens set to {max_new_tokens}")
                except ValueError:
                    print("Invalid token value. Use an integer.")
                continue
            elif cmd == "/penalty" and len(cmd_parts) > 1:
                try:
                    repetition_penalty = float(cmd_parts[1])
                    print(f"Repetition penalty set to {repetition_penalty}")
                except ValueError:
                    print("Invalid penalty value. Use a number between 1.0 and 2.0")
                continue
            elif cmd == "/format" and len(cmd_parts) > 1:
                format_style = cmd_parts[1].lower()
                if format_style not in ["instruction", "plain", "chat"]:
                    print("Invalid format. Choose: instruction, plain, or chat")
                else:
                    print(f"Format style set to {format_style}")
                continue
            elif cmd == "/system":
                if len(cmd_parts) > 1 and cmd_parts[1].lower() == "clear":
                    system_prompt = None
                    print("System prompt cleared.")
                elif len(cmd_parts) > 1:
                    # Join all parts after /system as the system prompt
                    system_prompt = " ".join(cmd_parts[1:])
                    print(f"System prompt set: {system_prompt[:60]}{'...' if len(system_prompt) > 60 else ''}")
                else:
                    if system_prompt:
                        print(f"Current system prompt: {system_prompt}")
                    else:
                        print("No system prompt set. Use: /system <your prompt>")
                continue
            else:
                print("Unknown command. Type /quit to exit.")
                continue

        # Format the prompt based on style
        if format_style == "chat":
            # Use proper chat format with system prompt and history
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add conversation history if enabled
            if use_history:
                for turn in conversation_history:
                    messages.append({"role": "user", "content": turn["user"]})
                    messages.append({"role": "assistant", "content": turn["assistant"]})

            messages.append({"role": "user", "content": user_input})
            formatted_prompt = format_chat(messages, tokenizer)
        elif format_style == "instruction":
            if use_history and conversation_history:
                # Build context from history
                history_text = ""
                for turn in conversation_history:
                    history_text += f"### Instruction:\n{turn['user']}\n\n### Response:\n{turn['assistant']}\n\n"
                formatted_prompt = f"{history_text}### Instruction:\n{user_input}\n\n### Response:\n"
            else:
                formatted_prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
        else:  # plain
            if use_history and conversation_history:
                # Build context from history
                history_text = ""
                for turn in conversation_history:
                    history_text += f"{turn['user']}\n{turn['assistant']}\n\n"
                formatted_prompt = f"{history_text}{user_input}"
            else:
                formatted_prompt = user_input

        # Generate response
        print("Assistant: ", end="", flush=True)
        response = generate_response(
            model,
            tokenizer,
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        print(response)
        print()

        # Save last response for /page command
        last_response = response

        # Update conversation history
        conversation_history.append({
            "user": user_input,
            "assistant": response
        })


def single_query(
    model,
    tokenizer,
    query: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    repetition_penalty: float = 1.2,
    format_style: str = "chat",
    system_prompt: str = "You are an assistant. Provide concise, direct answers."
):
    """
    Run a single query and return the response.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        query: The query text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        repetition_penalty: Penalty for repeating tokens
        format_style: How to format the prompt
        system_prompt: System prompt (used with chat format)
    """
    # Format the prompt
    if format_style == "chat":
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})
        formatted_prompt = format_chat(messages, tokenizer)
    elif format_style == "instruction":
        formatted_prompt = f"### Instruction:\n{query}\n\n### Response:\n"
    else:  # plain
        formatted_prompt = query

    # Generate and return response
    response = generate_response(
        model,
        tokenizer,
        formatted_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )

    return response


def main():
    parser = argparse.ArgumentParser(
        description="Interactive inference with finetuned models + LoRA (supports Qwen/Llama)"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        help="Path to LoRA adapter checkpoint (e.g., ./downloaded_models/custom_centrist/final_model). Leave empty to use base model only."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name (default: Qwen/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (requires more VRAM)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (default: 256)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["instruction", "plain", "chat"],
        default="chat",
        help="Prompt format style (default: chat - uses proper chat template with system prompt)"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are an assistant. Provide concise, direct answers.",
        help="System prompt to use with chat format (default: 'You are an assistant. Provide concise, direct answers.')"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query mode - provide a query and exit (non-interactive)"
    )
    parser.add_argument(
        "--use_history",
        action="store_true",
        help="Enable conversation history (stateful chat, default: disabled)"
    )

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        use_4bit=not args.no_4bit,
    )

    # Run in single query mode or interactive mode
    if args.query:
        response = single_query(
            model,
            tokenizer,
            args.query,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=1.2,
            format_style=args.format,
            system_prompt=args.system_prompt,
        )
        print(response)
    else:
        interactive_chat(
            model,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=1.2,
            format_style=args.format,
            system_prompt=args.system_prompt,
            use_history=args.use_history,
        )


if __name__ == "__main__":
    main()
