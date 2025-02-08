import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
import json
from PIL import Image
from typing import Dict, List, Optional
from dataclasses import dataclass
from bbox_tokenizer import BBoxTokenizer
from configuration_bbqwen2_vl import BBQwen2VLConfig
from processing_bbqwen2_vl import BBQwen2VLProcessor
from modeling_bbqwen2_vl import BBQwen2VLCausalLMOutputWithPast, BBQwen2VLForConditionalGeneration
import os

class BBoxDataset(Dataset):
    """Custom dataset for bounding box prediction"""
    def __init__(self, 
                 data_range: tuple,
                 data_dir: str,
                 processor: BBQwen2VLProcessor,
                 max_length: int = 512):
        self.processor = processor
        self.max_length = max_length
        
        # Load your dataset
        self.data_dir = data_dir
        with open(self.data_dir, 'r') as file:
            self.annotations = json.load(file)
        
        print(f"Loaded annotations size: {len(self.annotations)}")
        self.dataset = self.create_dataset_from_json(self.annotations, data_range)
        print(f"Final dataset size: {len(self.dataset)}")
        print(f"Data range used: {data_range}")

    def __len__(self):
        return len(self.dataset)  # Return length of processed dataset instead of annotations

    def __getitem__(self, idx):
        print(f"__getitem__ called with idx: {idx}")
        print(f"Current dataset length: {len(self.dataset)}")
        
        item = self.dataset[idx]
        
        print(f"bbox type: {type(item['bbox'])}")
        print(f"bbox value: {item['bbox']}")
        print(f"text value: {item['text']}")
        print(f"image size: {item['image'].size}")
        
        # The image is already loaded in the dataset
        image = item["image"]
        image_size = image.size  # Get width and height
        
        # Process image with processor's image processor
        processed_image = self.processor.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        
        # Process text and bboxes
        bbox_tokenizer = BBoxTokenizer(self.processor.tokenizer)
        text_outputs = bbox_tokenizer(
            text=item["text"],
            bbox=item["bbox"],
            image_size=image_size,
            padding=True,
            max_length=self.max_length
        )
        
        return {
            "pixel_values": processed_image,
            "input_ids": text_outputs["input_ids"],
            "attention_mask": text_outputs["attention_mask"],
            "bbox_tensors": text_outputs["bbox_tensors"]  # These will be used as bbox_labels
        }
    
    def create_dataset_from_json(self, json_data, data_range):
        dataset = []
        current_idx = 0

        for filename, info in json_data.items():
            sections = info["sections"]
            image_path = f"/home/tarch/datasets/synthetic/lines/english/{filename}.jpg"
        
            for section in sections:
                try:
                    image = Image.open(image_path).convert("RGB")
                except FileNotFoundError:
                    print(f"Warning: {filename} not found at {image_path}")
                    continue
                
                for paragraph in section["paragraphs"]:
                    for line in paragraph["lines"]:
                        if current_idx >= data_range[1]:
                            return dataset
                            
                        if current_idx >= data_range[0]:
                            # Add print to debug bbox format
                            print(f"Line bbox format: {line['bbox']}")
                            # Ensure bbox is a list of 4 values [x1, y1, x2, y2]
                            if not isinstance(line['bbox'], list):
                                print(f"Warning: Invalid bbox format for line: {line['bbox']}")
                                continue
                            
                            dataset.append({
                                "text": line["text"],
                                "image": image,
                                "bbox": line["bbox"]
                            })
                        current_idx += 1
                        
                        for word in line["words"]:
                            if current_idx >= data_range[1]:
                                return dataset
                                
                            if current_idx >= data_range[0]:
                                # Add print to debug bbox format
                                print(f"Word bbox format: {word['bbox']}")
                                # Ensure bbox is a list of 4 values [x1, y1, x2, y2]
                                if not isinstance(word['bbox'], list):
                                    print(f"Warning: Invalid bbox format for word: {word['bbox']}")
                                    continue
                                
                                dataset.append({
                                    "text": word["text"],
                                    "image": image,
                                    "bbox": word["bbox"]
                                })
                            current_idx += 1

        return dataset

@dataclass
class BBoxTrainingArguments(TrainingArguments):
    bbox_loss_weight: float = 1.0

class BBoxTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute training loss. Overrides Trainer.compute_loss.
        """
        print("COMPUTE LOSS CALLED\n")
        print("Inputs", inputs)
        print(f"pixel_values shape: {inputs['pixel_values'].shape}")
        
        # Handle both batched and unbatched inputs
        pixel_values = inputs["pixel_values"]
        if len(pixel_values.shape) == 3:
            # Add batch dimension if missing
            pixel_values = pixel_values.unsqueeze(0)
        
        batch_size, channels, height, width = pixel_values.shape
        
        # Calculate grid size
        grid_size = 32  # patch size
        grid_h = height // grid_size
        grid_w = width // grid_size
        
        # Create grid_thw as a tensor instead of list of tuples
        # Shape: [batch_size, 3] where each row is [t=1, h, w]
        image_grid_thw = torch.tensor(
            [[1, grid_h, grid_w]] * batch_size, 
            device=pixel_values.device,
            dtype=torch.long
        )
        
        # Ensure all inputs are batched
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        
        # Add labels for training
        labels = input_ids.clone()
        labels = torch.roll(labels, -1, dims=-1)  # Shift right by 1
        labels[:, -1] = -100  # Mask last token
        
        # Add bbox labels if available
        bbox_labels = inputs.get("bbox_tensors")
        if bbox_labels is not None and len(bbox_labels.shape) == 2:
            bbox_labels = bbox_labels.unsqueeze(0)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            bbox_labels=bbox_labels,
            image_grid_thw=image_grid_thw
        )
        
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

def main():
    # Initialize config and model
    config = BBQwen2VLConfig.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        bbox_size=4,  # for x1, y1, x2, y2
    )
    
    model = BBQwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        config=config,
    )
    
    # Initialize processor
    processor = BBQwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    # Create datasets
    train_dataset = BBoxDataset(
        data_range=(0,10),
        data_dir="/grphome/grp_handwriting/synthetic_data/lines/english_samples/OCR_9950000.json",
        processor=processor
    )
    # one bounding box per token
    eval_dataset = BBoxDataset(
        data_range=(11,21),
        data_dir="/grphome/grp_handwriting/synthetic_data/lines/english_samples/OCR_9950000.json",
        processor=processor
    )
    
    # Training arguments
    training_args = BBoxTrainingArguments(
        output_dir="/grphome/grp_handwriting/compute/bbox_qwen_output",
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
    trainer.save_model("/grphome/grp_handwriting/compute/bbox_qwen_final")

if __name__ == "__main__":
    main()