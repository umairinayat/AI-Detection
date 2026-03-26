---
language:
  - en
license: apache-2.0
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
  - peft
  - lora
  - qlora
  - transformers
  - text-classification
  - ai-detection
  - human-ai-detection
pipeline_tag: text-classification
library_name: peft
---

# Qwen2.5-3B AI vs Human text detector (QLoRA adapters)

This repository contains **QLoRA (LoRA) adapter weights** for binary **human vs AI-generated** text classification. The base model is **Qwen/Qwen2.5-3B-Instruct** loaded in 4-bit with **PEFT**; only adapters and tokenizer files are stored here.

## Intended use

- **Task:** Binary sequence classification: label `0` = human-written, `1` = AI-generated.
- **Input:** Plain English text (training focused on English; other languages may be less reliable).
- **Downstream:** Plug into the [Peft](https://github.com/huggingface/peft) + `transformers` stack for inference or further fine-tuning.

## Base model

- **Base:** [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)  
- **Head:** `Qwen2ForSequenceClassification` with 2 labels (see training config).

## Training summary

- **Method:** QLoRA (4-bit NF4, LoRA on attention projections `q_proj`, `k_proj`, `v_proj`, `o_proj`).
- **LoRA:** rank `16`, alpha `32`, dropout `0.05`; classifier head saved via `modules_to_save`.
- **Data:** Balanced mix from several public Hugging Face datasets (human/AI pairs and Wikipedia human text), subsampled and balanced to a large training split (~500K samples) with balanced eval.
- **Metrics:** Training used F1 for checkpoint selection; eval scores depend on the held-out split and are **not** a guarantee of out-of-domain performance.

## Limitations

- Performance is **dataset-dependent**; strong in-distribution accuracy does not imply robustness to new domains, languages, or adversarial rewrites.
- **Do not** use as sole evidence for high-stakes decisions (academic integrity, employment, legal).
- Biases and errors in training data can affect predictions.

## How to load (inference)

```python
import torch
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "<YOUR_HF_USERNAME>/llm_detector"  # this repo

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=2, quantization_config=bnb, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(base, ADAPTER)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

## License

Adapter weights follow the licensing terms of the upstream **Qwen2.5** model and this project’s release policy. Use **Qwen** license terms and **Apache-2.0** where applicable; refer to the base model card for the authoritative license text.

## Citation

If you use these adapters, cite the **Qwen2.5** model and **PEFT** as appropriate:

```bibtex
@misc{qwen2_5,
  title={Qwen2.5},
  howpublished={\url{https://huggingface.co/Qwen}},
}
```
