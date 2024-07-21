import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


if torch.cuda.is_available():
    device = torch.device("cuda")
elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
    and torch.backends.mps.is_built()
):
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"running device: {device}")
auth_token = os.environ.get("TOKEN_READ_SECRET") or True

tokenizer = AutoTokenizer.from_pretrained(
    "NorHsangPha/shan_gpt2_news", token=auth_token
)
model = AutoModelForCausalLM.from_pretrained(
    "NorHsangPha/shan_gpt2_news", pad_token_id=tokenizer.eos_token_id, token=auth_token
).to(device)


def greedy_search(model_inputs, max_new_tokens):
    greedy_output = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    return tokenizer.decode(greedy_output[0], skip_special_tokens=True)


def beem_search(model_inputs, max_new_tokens):
    beam_output = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        num_beams=5,
        no_repeat_ngram_size=2,  #
        num_return_sequences=5,  #
        early_stopping=True,
    )

    return tokenizer.decode(beam_output[0], skip_special_tokens=True)


def sample_outputs(model_inputs, max_new_tokens):
    sample_output = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=0,
        temperature=0.6,
    )

    return tokenizer.decode(sample_output[0], skip_special_tokens=True)


def top_k_search(model_inputs, max_new_tokens):
    top_k_output = model.generate(
        **model_inputs, max_new_tokens=max_new_tokens, do_sample=True, top_k=50
    )

    return tokenizer.decode(top_k_output[0], skip_special_tokens=True)


def top_p_search(model_inputs, max_new_tokens):
    top_p_output = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.92,
        top_k=0,
    )

    return tokenizer.decode(top_p_output[0], skip_special_tokens=True)


def generate_text(input_text, search_method="sample_outputs"):
    model_inputs = tokenizer(input_text, return_tensors="pt").to(device)
    max_new_tokens = 120

    match search_method:
        case "greedy_search":
            text = greedy_search(model_inputs, max_new_tokens)

        case "beem_search":
            text = beem_search(model_inputs, max_new_tokens)

        case "top_k_search":
            text = top_k_search(model_inputs, max_new_tokens)

        case "top_p_search":
            text = top_p_search(model_inputs, max_new_tokens)

        case _:
            text = sample_outputs(model_inputs, max_new_tokens)

    return text


GENERATE_EXAMPLES = [
    ["မႂ်ႇသုင်ၶႃႈ", "sample_outputs"],
    ["ပၢင်တိုၵ်းသိုၵ်းသိူဝ်", "greedy_search"],
    ["ပၢင်တိုၵ်းသိုၵ်းသိူဝ်", "top_k_search"],
    ["ပၢင်တိုၵ်းသိုၵ်းသိူဝ်", "top_p_search"],
    ["ပၢင်တိုၵ်းသိုၵ်းသိူဝ်", "beem_search"],
]
