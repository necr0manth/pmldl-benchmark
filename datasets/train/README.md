# Robot-Guide VLM Training Dataset (`datasets/train`)

Complete multimodal training dataset constructed from 110 museum photography frames (`finetune-people-collected`) designed for supervised fine-tuning (SFT) and parameter-efficient fine-tuning (LoRA / QLoRA) of vision-language models (e.g., Qwen2.5-VL, Qwen3.5, InternVL, Gemma).

---

## 1. Dataset Overview and Composition

- **Split**: `train` (guaranteed strictly disjoint `group_id` sets; passes cross-split leakage checks against `dev`).
- **Total Image Assets**: 110 photographs (`datasets/train/assets/ft_a_01.png` to `ft_f_028.png`).
- **Benchmark Cases**: 330 cases across three callable methods.
- **SFT Conversations**: 330 multi-turn vision-language conversation pairs.

### Category and Action Breakdown

| Category | Images | Target `check_people` Action | Description |
|---|---:|:---:|---|
| **`attentive`** | 77 | `idle` | 1 or more visitors facing the robot camera directly with attentive gaze. |
| **`inattentive`** | 16 | `interrupt` | Visitors present with backs turned, in profile, or attending to wall displays. |
| **`empty`** | 17 | `interrupt` | Empty museum corridors, galleries, and exhibit halls with zero visitors. |
| **Total** | **110** | — | **77 `idle` / 33 `interrupt`** |

---

## 2. Directory Layout

```
datasets/train/
├── assets/                  # 110 full-resolution PNG images
│   ├── ft_a_01.png ... ft_a_10.png
│   ├── ft_b_01.png ... ft_b_10.png
│   ├── ft_c_001.png ... ft_c_022.png
│   ├── ft_d_001.png ... ft_d_020.png
│   ├── ft_e_001.png ... ft_e_020.png
│   └── ft_f_001.png ... ft_f_028.png
├── cases.jsonl              # 330 benchmark cases (split: "train")
├── manifest.json            # Dataset manifest with SHA-256 checksums and statistics
├── sft_qwen.jsonl           # SFT dataset in OpenAI/Unsloth messages format
├── sft_sharegpt.json        # SFT dataset in ShareGPT conversations format
└── README.md                # Documentation and fine-tuning guide
```

---

## 3. Case Types in `cases.jsonl` (330 Cases)

To prevent catastrophic forgetting of structured JSON routing when fine-tuning on perception alone, the training dataset covers all three system methods:

1. **`check_people` (110 cases)**:
   - Pure image-only audience attentiveness decision.
   - Target schema: `{"tool": "idle" | "interrupt", "args": {}, "abstain": false}`.
   - Evaluates whether the robot should continue narrating (`idle`) or pause (`interrupt`).
2. **`describe_image` (110 cases)**:
   - Pure image-only scene narration.
   - Target schema: `{"kind": "nonempty_text"}`.
   - High-fidelity grounding on real exhibit features (armatures, pottery, paintings, corridors).
3. **`decide` (110 cases)**:
   - Full multimodal decision contract (transcript + image + mission state + task policy).
   - Balanced distribution of robot navigation commands (`goto_exhibit`), tour launches (`start_tour`), and canonical abstentions (`abstain: true`).

---

## 4. Ready-to-Use SFT Training Formats

### Format A: `sft_qwen.jsonl` (Hugging Face / Unsloth format)
Compatible with Hugging Face `trl.SFTTrainer`, `transformers`, and `unsloth`:
```json
{
  "id": "train-people-ft_a_01",
  "task": "check_people",
  "images": ["assets/ft_a_01.png"],
  "messages": [
    {
      "role": "system",
      "content": "You are an audience-engagement sensor on a mobile robot guide..."
    },
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "assets/ft_a_01.png"},
        {"type": "text", "text": "Определи статус внимания посетителей перед роботом."}
      ]
    },
    {
      "role": "assistant",
      "content": "{\"tool\": \"idle\", \"args\": {}, \"abstain\": false}"
    }
  ]
}
```

### Format B: `sft_sharegpt.json` (LLaMA-Factory / Axolotl format)
Compatible with LLaMA-Factory multi-modal pipelines:
```json
[
  {
    "id": "train-people-ft_a_01",
    "image": "assets/ft_a_01.png",
    "conversations": [
      {"from": "system", "value": "..."},
      {"from": "user", "value": "<image>\nОпредели статус внимания посетителей перед роботом."},
      {"from": "assistant", "value": "{\"tool\": \"idle\", \"args\": {}, \"abstain\": false}"}
    ]
  }
]
```

---

## 5. Verification and Validation

Verify dataset syntax and schema compliance:
```powershell
python -m vlm_benchmark validate .\datasets\train\cases.jsonl
```

Verify zero split leakage against the development benchmark (`dev` vs `train`):
```powershell
python -m vlm_benchmark validate .\datasets\methods\cases.jsonl .\datasets\train\cases.jsonl
```

Execute a dry run through the benchmark engine:
```powershell
python -m vlm_benchmark run .\datasets\train\cases.jsonl `
  --config .\configs\v2-mock.json `
  --output .\runs\train-mock
```

---

## 6. How to Fine-Tune Qwen with LoRA (Unsloth Example)

Below is an example training script using Unsloth for Qwen2.5-VL / Qwen3.5:

```python
import json
from datasets import Dataset
from unsloth import FastVisionModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig

# 1. Load model with 4-bit quantization
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-3B-Instruct",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# 2. Add LoRA adapters
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,  # Fine-tune visual encoder layers
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    random_state=3407,
)

# 3. Load SFT dataset
with open("datasets/train/sft_qwen.jsonl", "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

dataset = Dataset.from_list(raw_data)

# 4. Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="messages",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        output_dir="outputs/qwen-lora",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
    ),
)
trainer.train()

# 5. Save LoRA weights or export to GGUF
model.save_pretrained_merged("models/qwen-robot-finetuned", tokenizer, save_method="merged_16bit")
```

---

## 7. Rebuilding the Dataset

The complete training dataset can be regenerated deterministically at any time using:
```powershell
python scripts/build_train_dataset.py
```
This reads from `finetune-people-collected/manifest.jsonl`, synchronizes the assets into `datasets/train/assets/`, recalculates SHA-256 hashes, and rebuilds `cases.jsonl`, `sft_qwen.jsonl`, `sft_sharegpt.json`, and `manifest.json`.
