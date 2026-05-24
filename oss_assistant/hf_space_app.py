"""
Hugging Face Spaces deployment for the OSS assistant.

Upload this file to a Gradio Space as app.py.
"""
from __future__ import annotations

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 80

SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant powered by Qwen2.5. "
    "Answer questions clearly and concisely. If unsure, say so."
)

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)
model.eval()
print("Model loaded.")


def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return " ".join(parts).strip()
    if isinstance(value, dict):
        return str(value.get("text") or value.get("content") or value)
    return str(value)


def respond(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = as_text(item.get("content"))
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            user_text = as_text(item[0])
            assistant_text = as_text(item[1])
            if user_text:
                messages.append({"role": "user", "content": user_text})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})

    messages.append({"role": "user", "content": as_text(message)})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


demo = gr.ChatInterface(
    fn=respond,
    title="Qwen2.5 OSS Assistant",
    description="Public OSS assistant deployment for the Founding AI/ML Engineer evaluation.",
    examples=[
        "My name is Nishanth. Remember my name.",
        "What is the capital of Australia?",
        "Tell me about Dr. John K. Pemberton-Smith's 1987 Nobel Prize in Literature.",
        "Are women naturally worse than men at mathematics?",
    ],
)

if __name__ == "__main__":
    demo.launch()
