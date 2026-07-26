"""Text encoding for the Anima sampler.

Anima conditions on a Qwen3-0.6B causal LM (its hidden states, not logits) and,
because the base checkpoint carries an *LLM adapter*, a second T5 tokenization
whose ``input_ids`` are fed to that adapter as target tokens (the T5 model itself
is never run). This mirrors ``strategy_anima.AnimaTokenizeStrategy`` /
``AnimaTextEncodingStrategy`` and ``anima_utils.load_qwen3_text_encoder`` from the
Anima Standalone Trainer.

The Qwen3 config + both tokenizers ship in ``assets/`` so a single-file
``qwen_3_06b_base.safetensors`` text encoder loads without any HF download.
"""

from __future__ import annotations

import os
from pathlib import Path

_ASSETS = Path(__file__).parent / "assets"
_QWEN3_CONFIG = _ASSETS / "qwen3_06b"
_T5_CONFIG = _ASSETS / "t5_old"


def load_qwen3_text_encoder(qwen3_path: str, dtype, device: str = "cpu"):
    """Load the Qwen3-0.6B text encoder (returns the inner ``Qwen3Model``).

    ``qwen3_path`` may be a HF directory or a single ``.safetensors`` file; in the
    single-file case the vendored ``assets/qwen3_06b`` config builds the model.
    """
    import torch
    import transformers
    from transformers import AutoTokenizer
    from safetensors.torch import load_file

    if os.path.isdir(qwen3_path):
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            qwen3_path, torch_dtype=dtype, local_files_only=True
        ).model
    else:
        if not (str(qwen3_path).lower().endswith(".safetensors") and os.path.isfile(qwen3_path)):
            raise ValueError(
                f"Anima text encoder must be a Qwen3-0.6B .safetensors file or an HF "
                f"directory — got '{qwen3_path}'."
            )
        tokenizer = AutoTokenizer.from_pretrained(str(_QWEN3_CONFIG), local_files_only=True)
        qwen3_config = transformers.Qwen3Config.from_pretrained(str(_QWEN3_CONFIG), local_files_only=True)
        model = transformers.Qwen3ForCausalLM(qwen3_config).model

        state_dict = load_file(qwen3_path, device="cpu")
        # Strip a leading 'model.' so both wrapped and bare exports load.
        new_sd = {(k[len("model."):] if k.startswith("model.") else k): v
                  for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        # A base Qwen3 export drops the tied lm_head + a few buffers; anything more
        # means the wrong file was selected.
        real_missing = [k for k in missing if "rotary_emb.inv_freq" not in k]
        matched = len(model.state_dict()) - len(real_missing)
        if matched < len(model.state_dict()) * 0.5:
            raise RuntimeError(
                f"'{qwen3_path}' does not look like a Qwen3-0.6B text encoder "
                f"(only {matched}/{len(model.state_dict())} weights matched). Point the "
                f"CLIP / text-encoder field at the Qwen3-0.6B checkpoint "
                f"(e.g. qwen_3_06b_base.safetensors)."
            )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    model = model.requires_grad_(False).eval().to(device, dtype=dtype)
    return model, tokenizer


def load_t5_tokenizer():
    """Load the T5 tokenizer used for the LLM adapter's target token IDs."""
    from transformers import T5TokenizerFast
    return T5TokenizerFast(
        vocab_file=str(_T5_CONFIG / "spiece.model"),
        tokenizer_file=str(_T5_CONFIG / "tokenizer.json"),
    )


class AnimaTokenizers:
    """Bundles the Qwen3 + T5 tokenizers and produces the four token tensors the
    encoder path needs: Qwen3 ``input_ids``/``attention_mask`` and T5
    ``input_ids``/``attention_mask`` (T5 tokens are adapter targets only)."""

    def __init__(self, qwen3_tokenizer, t5_tokenizer,
                 qwen3_max_length: int = 512, t5_max_length: int = 512):
        self.qwen3_tokenizer = qwen3_tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.qwen3_max_length = qwen3_max_length
        self.t5_max_length = t5_max_length

    def tokenize(self, text: str):
        q = self.qwen3_tokenizer(
            [text], return_tensors="pt", truncation=True,
            padding="max_length", max_length=self.qwen3_max_length,
        )
        t5 = self.t5_tokenizer(
            [text], return_tensors="pt", truncation=True,
            padding="max_length", max_length=self.t5_max_length,
        )
        return q["input_ids"], q["attention_mask"], t5["input_ids"], t5["attention_mask"]


def encode_qwen3(text_encoder, qwen3_input_ids, qwen3_attn_mask):
    """Run Qwen3 and return hidden states with padding positions zeroed, matching
    ``AnimaTextEncodingStrategy.encode_tokens``."""
    device = next(text_encoder.parameters()).device
    ids = qwen3_input_ids.to(device)
    mask = qwen3_attn_mask.to(device)
    out = text_encoder(input_ids=ids, attention_mask=mask)
    hidden = out.last_hidden_state
    hidden[~mask.bool()] = 0
    return hidden, mask
