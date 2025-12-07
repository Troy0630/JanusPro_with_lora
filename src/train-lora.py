
import argparse
import json
import logging
import os
import random
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, get_scheduler

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import matplotlib.pyplot as plt
from transformers import TrainerCallback
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LossPlotCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.losses = []
        self.output_dir = output_dir

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            self.losses.append(logs['loss'])
            # Save the loss plot periodically or at the end of training
            if len(self.losses) % 50 == 0 or state.global_step == state.max_steps:
                self.plot_loss()

    def plot_loss(self):
        """
        Plot and save the loss curve.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, label='Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'training_loss_curve.png'))
        plt.close()

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Multi-modal fine-tuning training script")
    parser.add_argument(
        "--json_path", type=str, required=True, help="Path to JSON annotation file"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Directory containing images"
    )
    parser.add_argument("--pretrained_model", type=str, default="Janus-Pro-1B", help="Pretrained model identifier")
    parser.add_argument(
        "--output_dir", type=str, default="./janus_lora_output", help="Output directory for fine-tuned model"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device for training")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def set_seed(seed: int):
    """
    Set random seed to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def load_json_data(json_path: str, image_dir: str) -> List[Tuple[str, str]]:
    """
    Load image-text pairs from JSON file and image directory.

    Args:
        json_path (str): Path to JSON annotation file.
        image_dir (str): Directory containing images.

    Returns:
        List[Tuple[str, str]]: A list of (image_path, caption) pairs.

    Raises:
        ValueError: If no valid image-text pairs are found.
    """
    pairs = []
    
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if annotations exist in JSON
    if 'annotations' not in data:
        raise ValueError("JSON file must contain 'annotations' field")
    
    # Build pairs
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        
        # Find image file (support multiple extensions)
        img_exts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        img_path = None
        
        for ext in img_exts:
            possible_path = os.path.join(image_dir, f"{image_id}{ext}")
            if os.path.exists(possible_path):
                img_path = possible_path
                break
        
        if img_path is not None:
            pairs.append((img_path, caption))
        else:
            logger.warning(f"No image found for image_id: {image_id}")
    
    if not pairs:
        raise ValueError(f"No valid image-text pairs found in {json_path} and {image_dir}")
    
    logger.info(f"Loaded {len(pairs)} samples from {json_path}")
    return pairs


class MultiModalTrainDataset(Dataset):
    """
    Custom dataset for multi-modal training.
    """

    def __init__(self, pairs: List[Tuple[str, str]]):
        self.data = pairs

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, txt = self.data[idx]
        return {"image_path": img_path, "text": txt}


def conversation_template(
    image_path: str, user_text: str = "通过这张胸部x光可以诊断出什么?", assistant_text: str = ""
) -> List[Dict[str, Any]]:
    """
    Construct a conversation template.

    Args:
        image_path (str): Path to the image.
        user_text (str): User's input prompt.
        assistant_text (str): Assistant's response.

    Returns:
        List[Dict[str, Any]]: List of conversation turns.
    """
    return [
        {"role": "<|User|>", "content": f"<image_placeholder>\n{user_text}", "images": [image_path]},
        {"role": "<|Assistant|>", "content": assistant_text},
    ]


def collate_fn_for_vlchat(batch: List[Dict[str, Any]], processor: VLChatProcessor) -> Dict[str, torch.Tensor]:
    """
    Collate function for VLChat processing.

    Args:
        batch (List[Dict[str, Any]]): A batch of samples.
        processor (VLChatProcessor): Pretrained processor.

    Returns:
        Dict[str, torch.Tensor]: Processed batch with input tensors.
    """
    conversations = []
    for item in batch:
        conversations.extend(conversation_template(image_path=item["image_path"], assistant_text=item["text"]))

    pil_images_list = load_pil_images(conversations)
    encoded = processor(
        conversations=conversations,
        images=pil_images_list,
        return_tensors="pt",
        force_batchify=True,
    )
    encoded["labels"] = encoded["input_ids"].clone()
    return dict(encoded)


def new_forward(
    self,
    input_ids=None,
    attention_mask=None,
    pixel_values=None,
    images_seq_mask=None,
    images_emb_mask=None,
    labels=None,
    **kwargs,
):
    """
    Custom forward method to process image inputs.
    """
    inputs_embeds = None
    if pixel_values is not None:
        inputs_embeds = self.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
        )

    if inputs_embeds is not None:
        kwargs.pop("inputs_embeds", None)
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
    else:
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
    return outputs


# Override the forward method of MultiModalityCausalLM
MultiModalityCausalLM.forward = new_forward


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load dataset
    pairs = load_json_data(args.json_path, args.image_dir)
    dataset = MultiModalTrainDataset(pairs)

    # Debugging: Check the first few samples
    for i, sample in enumerate(dataset):
        if i >= 5:
            break
        logger.debug(f"Sample {i}: {sample}")

    # Load processor and pretrained model
    logger.info(f"Loading pretrained model: {args.pretrained_model}")
    processor = VLChatProcessor.from_pretrained(args.pretrained_model)
    model = MultiModalityCausalLM.from_pretrained(args.pretrained_model, torch_dtype=torch.float16, device_map="auto")
    
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"params: {total_params}")
    # Configure LoRA for efficient fine-tuning
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable params: {trainable_params} / {total_params}")

    # Prepare optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    num_training_steps = len(dataset) * args.max_epochs // args.batch_size
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warm-up

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=True,
        max_grad_norm=1.0,
        save_strategy="epoch",
        evaluation_strategy="no",
        logging_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    # Initialize Hugging Face Trainer with the LossPlotCallback
    loss_plot_callback = LossPlotCallback(output_dir=args.output_dir)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=partial(collate_fn_for_vlchat, processor=processor),
        optimizers=(optimizer, lr_scheduler),
        callbacks=[loss_plot_callback]  # Add the callback here
    )

    logger.info("Starting training...")
    trainer.train()

    # Save fine-tuned model and processor
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    logger.info(f"Fine-tuned model saved to {args.output_dir}")



if __name__ == "__main__":
    main()