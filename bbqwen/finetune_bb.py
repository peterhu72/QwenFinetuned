import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
import json
from PIL import Image
from typing import Dict, List, Optional
from dataclasses import dataclass
import os

class BBoxDataset(Dataset):
    """Custom dataset for bounding box prediction"""
    def __init__(self, 
                 data_dir: str,
                 processor: BBQwen2VLProcessor,
                 max_length: int = 512):
        self.processor = processor
        self.max_length = max_length
        
        # Load your dataset
        # Assuming format: {"image_path": str, "text": str, "bbox": [x1,y1,x2,y2]}
        with open(os.path.join(data_dir, "/grphome/grp_handwriting/synthetic_data/lines/english_samples/OCR_9950000.json"), "r") as f:
            self.annotations = json.load(f)
        self.data_dir = data_dir
        self.dataset = []

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        
        # Load and preprocess image
        image = Image.open(os.path.join(self.data_dir, item["image_path"])).convert("RGB")
        
        # Create prompt template
        # Modify this based on your specific use case
        prompt = f"Find the bounding box for the text: {item['text']}"
        
        # Process inputs
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            bboxes=[item["bbox"]],  # Pass bounding box coordinates
            return_bbox_labels=True
        )
        
        # Remove batch dimension added by processor
        for k in inputs:
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].squeeze(0)
        
        return inputs
    
    def create_dataset_from_json(json_data):
        dataset = []
        count = 0

        for filename, info in json_data.items():
            
            sections = info["sections"]  # Extract the text as the label
            image_path = f"/home/tarch/datasets/synthetic/lines/english/{filename}.jpg"  # Construct the full path
            
        
            for section in sections:
                try:
                    
                    # Load and convert the image to RGB format
                    image = Image.open(image_path).convert("RGB")
                    #processed_image = transform_image(image)
                
                except FileNotFoundError:
                    print(f"Warning: {filename} not found at {image_path}")
                    continue
                
                for paragraph in section["paragraphs"]:
                    
                    for line in paragraph["lines"]:
                    
                        dataset.append({
                            "text": line["text"],
                            "image": image,
                            "bbox": line["bbox"]
                        })
                        
                        for word in line["words"]:
                            dataset.append({
                                "text": word["text"],
                                "image": image,
                                "bbox": word["bbox"]
                            })

        return dataset

@dataclass
class BBoxTrainingArguments(TrainingArguments):
    bbox_loss_weight: float = 1.0

class BBoxTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            labels=inputs.get("labels"),
            bbox_labels=inputs.get("bbox_labels")
        )
        
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

def main():
    # Initialize config and model
    config = BBQwen2VLConfig.from_pretrained(
        "Qwen/Qwen2-VL-7B",
        bbox_size=4,  # for x1, y1, x2, y2
    )
    
    model = BBQwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B",
        config=config,
    )
    
    # Initialize processor
    processor = BBQwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B")
    
    # Create datasets
    train_dataset = BBoxDataset(
        data_dir="path/to/train/data",
        processor=processor
    )
    
    eval_dataset = BBoxDataset(
        data_dir="path/to/eval/data",
        processor=processor
    )
    
    # Training arguments
    training_args = BBoxTrainingArguments(
        output_dir="./bbox_qwen_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=2e-5,
        bbox_loss_weight=1.0,
        fp16=True,  # Enable mixed precision training
    )
    
    # Initialize trainer
    trainer = BBoxTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Start training
    trainer.train()
    
    # Save the final model
    trainer.save_model("./bbox_qwen_final")

if __name__ == "__main__":
    main()