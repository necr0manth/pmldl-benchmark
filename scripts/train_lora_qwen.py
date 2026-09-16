"""Fine-tune Qwen3.5-4B using LoRA / QLoRA with the optimal few-shot prompt."""

from __future__ import annotations

import argparse
import json
import logging
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qwen_finetune")

ROOT = Path(__file__).resolve().parents[1]


def load_cases(dataset_path: Path, methods_filter: list[str] | None = None) -> list[dict[str, Any]]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    cases = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        if methods_filter and c.get("method") not in methods_filter:
            continue
        cases.append(c)
    return cases


class QwenFewShotVLMDataset(torch.utils.data.Dataset):
    """PyTorch Dataset that formats cases with the optimal few-shot prompt."""

    def __init__(
        self,
        cases: list[dict[str, Any]],
        dataset_root: Path,
        processor: Any,
        people_prompt: str,
        desc_prompt: str,
        decision_prompt_tmpl: str,
    ) -> None:
        self.cases = cases
        self.dataset_root = dataset_root
        self.processor = processor
        self.people_prompt = people_prompt
        self.desc_prompt = desc_prompt
        self.decision_prompt_tmpl = decision_prompt_tmpl

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        case = self.cases[idx]
        method = case.get("method", "decide")

        # 1. Load image
        img_rel = case["image"]["path"]
        img_path = (self.dataset_root / img_rel).resolve()
        if not img_path.is_file():
            raise FileNotFoundError(f"Case image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")

        # 2. Build multi-turn messages with the best few-shot prompt
        if method == "check_people":
            system_prompt = self.people_prompt
            user_text = "Определи статус внимания посетителей перед роботом."
            resp_obj = dict(case["expected"])
            if "confidence" not in resp_obj:
                resp_obj["confidence"] = 0.0 if resp_obj.get("abstain") else 0.95
            ordered_keys = ["tool", "args", "confidence", "abstain"]
            ordered_obj = {k: resp_obj[k] for k in ordered_keys if k in resp_obj}
            for k, v in resp_obj.items():
                if k not in ordered_obj:
                    ordered_obj[k] = v
            assistant_content = json.dumps(ordered_obj, ensure_ascii=False)
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_text},
                    ],
                },
                {"role": "assistant", "content": assistant_content},
            ]
        elif method == "describe_image":
            system_prompt = self.desc_prompt
            user_text = "Опиши сцену перед роботом-экскурсоводом."
            notes = case.get("metadata", {}).get("notes") or ""
            assistant_content = f"В музейном зале: {notes}" if notes else "Музейная экспозиция с экспонатами и галерейным пространством."
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_text},
                    ],
                },
                {"role": "assistant", "content": assistant_content},
            ]
        else:  # decide
            transcript = case.get("transcript", "")
            mission_state = json.dumps(case.get("mission_state", {}), ensure_ascii=False)
            extra_context = case.get("extra_context", "")
            prompt_content = (
                self.decision_prompt_tmpl
                .replace("{transcript}", transcript)
                .replace("{{TRANSCRIPT_JSON}}", json.dumps(transcript, ensure_ascii=False))
                .replace("{image_status}", "available")
                .replace("{{IMAGE_STATUS_JSON}}", '"available"')
                .replace("{mission_state}", mission_state)
                .replace("{{MISSION_STATE_JSON}}", mission_state)
                .replace("{extra_context}", extra_context)
                .replace("{{EXTRA_CONTEXT}}", extra_context)
            )
            resp_obj = dict(case["expected"])
            if "confidence" not in resp_obj:
                resp_obj["confidence"] = 0.0 if resp_obj.get("abstain") else 0.95
            ordered_keys = ["tool", "args", "confidence", "abstain"]
            ordered_obj = {k: resp_obj[k] for k in ordered_keys if k in resp_obj}
            for k, v in resp_obj.items():
                if k not in ordered_obj:
                    ordered_obj[k] = v
            assistant_content = json.dumps(ordered_obj, ensure_ascii=False)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_content},
                    ],
                },
                {"role": "assistant", "content": assistant_content},
            ]

        # 3. Format text via chat template
        full_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # 4. Tokenize image and text
        inputs = self.processor(text=[full_text], images=[image], return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        # 5. Mask prompt tokens from loss calculation (loss only on assistant output)
        labels = input_ids.clone()
        match_idx = -1
        for candidate_header in ("</think>\n\n", "<|im_start|>assistant\n"):
            header_bytes = self.processor.tokenizer.encode(candidate_header, add_special_tokens=False)
            for i in range(len(input_ids) - len(header_bytes) + 1):
                if input_ids[i : i + len(header_bytes)].tolist() == header_bytes:
                    match_idx = i + len(header_bytes)
                    break
            if match_idx != -1:
                break

        if match_idx != -1:
            labels[:match_idx] = -100

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if "mm_token_type_ids" in inputs and inputs["mm_token_type_ids"] is not None:
            sample["mm_token_type_ids"] = inputs["mm_token_type_ids"][0]
        for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            if k in inputs and inputs[k] is not None:
                sample[k] = inputs[k]

        return sample


class QwenVLMCollator:
    """Collates and dynamically pads multimodal batches for Qwen."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )

        batch_out = {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask_padded,
        }

        if "mm_token_type_ids" in batch[0]:
            mm_ids = [item["mm_token_type_ids"] for item in batch]
            batch_out["mm_token_type_ids"] = torch.nn.utils.rnn.pad_sequence(
                mm_ids, batch_first=True, padding_value=0
            )

        # Collate image features if present
        for key in ("pixel_values", "image_grid_thw"):
            if key in batch[0]:
                tensors = [item[key] for item in batch]
                if all(isinstance(t, torch.Tensor) for t in tensors):
                    batch_out[key] = torch.cat(tensors, dim=0)

        return batch_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3.5-4B with Few-Shot Prompt using LoRA/QLoRA")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-4B", help="Hugging Face model ID or path")
    parser.add_argument("--train_dataset", type=str, default="datasets/train/cases.jsonl", help="Path to train cases.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen3.5-4b-lora", help="Output directory for LoRA adapter")
    parser.add_argument("--people_prompt_path", type=str, default="prompts/people-fewshot.txt", help="Path to few-shot people prompt")
    parser.add_argument("--decision_prompt_path", type=str, default="prompts/decision-v2.txt", help="Path to decide prompt")
    parser.add_argument("--description_prompt_path", type=str, default="prompts/description-system.txt", help="Path to description prompt")
    parser.add_argument("--method_filter", type=str, default="all", help="Comma-separated methods to include: check_people,decide,describe_image or all")
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "8bit", "none"], help="QLoRA quantization type")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--merge_and_save", action="store_true", help="Merge LoRA weights with base model after training")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (overrides epochs if > 0)")
    parser.add_argument("--dry_run", action="store_true", help="Parse dataset, verify templates and collate without training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting Qwen3.5-4B Fine-Tuning Setup")
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Few-Shot People Prompt: {args.people_prompt_path}")

    # Load prompt texts
    people_prompt = (ROOT / args.people_prompt_path).read_text(encoding="utf-8").strip()
    decision_prompt = (ROOT / args.decision_prompt_path).read_text(encoding="utf-8").strip()
    desc_prompt = (ROOT / args.description_prompt_path).read_text(encoding="utf-8").strip()

    # Load dataset cases
    dataset_file = (ROOT / args.train_dataset).resolve()
    methods_filter = None if args.method_filter == "all" else [m.strip() for m in args.method_filter.split(",")]
    cases = load_cases(dataset_file, methods_filter)
    logger.info(f"Loaded {len(cases)} training cases (filter={args.method_filter})")

    # Load processor
    from transformers import AutoProcessor
    logger.info("Loading AutoProcessor...")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    # Initialize PyTorch Dataset
    dataset = QwenFewShotVLMDataset(
        cases=cases,
        dataset_root=dataset_file.parent,
        processor=processor,
        people_prompt=people_prompt,
        desc_prompt=desc_prompt,
        decision_prompt_tmpl=decision_prompt,
    )

    if args.dry_run:
        logger.info("Dry run mode requested: verifying samples...")
        sample_0 = dataset[0]
        logger.info(f"Sample 0 input_ids shape: {sample_0['input_ids'].shape}")
        logger.info(f"Sample 0 active labels: {(sample_0['labels'] != -100).sum().item()} tokens")
        collator = QwenVLMCollator(pad_token_id=processor.tokenizer.pad_token_id or 0)
        batch = collator([dataset[0], dataset[min(1, len(dataset) - 1)]])
        logger.info(f"Batched input_ids shape: {batch['input_ids'].shape}")
        logger.info("Dry run check PASSED. Exiting without model training.")
        return

    # Load PEFT and Quantization libraries
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig, Trainer, TrainingArguments

    # Configure Quantization
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quant_config = None
    if args.quantization == "4bit":
        logger.info(f"Enabling 4-bit QLoRA (NF4, compute_dtype={compute_dtype})")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif args.quantization == "8bit":
        logger.info("Enabling 8-bit quantization")
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load base model
    logger.info("Loading base vision-language model...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        quantization_config=quant_config,
        torch_dtype=compute_dtype if args.quantization == "none" else None,
        device_map="auto",
    )

    if args.quantization in ("4bit", "8bit"):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Setup LoRA
    logger.info(f"Configuring LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training arguments
    out_path = Path(args.output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    save_strat = "steps" if args.max_steps > 0 else "epoch"
    save_st = args.max_steps if args.max_steps > 0 else 500

    training_args = TrainingArguments(
        output_dir=str(out_path),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy=save_strat,
        save_steps=save_st,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        dataloader_num_workers=0,
        report_to="none",
    )

    collator = QwenVLMCollator(pad_token_id=processor.tokenizer.pad_token_id or 0)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    logger.info("Starting training loop...")
    train_result = trainer.train()
    logger.info(f"Training completed! Loss: {train_result.training_loss:.4f}")

    # Save LoRA adapter
    logger.info(f"Saving fine-tuned LoRA adapter to {out_path}...")
    model.save_pretrained(str(out_path))
    processor.save_pretrained(str(out_path))

    # Optional merge
    if args.merge_and_save:
        merged_dir = out_path.parent / f"{out_path.name}-merged"
        logger.info(f"Merging LoRA adapter with base model into {merged_dir}...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir))
        processor.save_pretrained(str(merged_dir))
        logger.info(f"Merged model saved successfully at {merged_dir}")

    logger.info("All operations complete successfully!")


if __name__ == "__main__":
    main()
