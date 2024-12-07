import torch
import json
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLConfig,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from torchvision import transforms
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

'''

This File is for finetuning Qwen in order to predict bounding box coordinates. 
Important to note:
We are resizing the images to 224 x 224 for finetuning
etc...

'''

def resize_and_get_scales(image):
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height
        
        if aspect_ratio > 1:
            # Width is larger
            new_width = max_dim
            new_height = int(new_width / aspect_ratio)
        else:
            # Height is larger
            new_height = max_dim
            new_width = int(new_height * aspect_ratio)
            
        # Ensure dimensions are divisible by patch_size
        new_width = (new_width // patch_size) * patch_size
        new_height = (new_height // patch_size) * patch_size
        
        # Calculate scaling factors
        width_scale = original_width / new_width
        height_scale = original_height / new_height
        
        # Resize image
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image, (width_scale, height_scale)

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image)

def create_dataset_from_json(json_data):
    dataset = []
    count = 0

    for filename, info in json_data.items():
        
        count += 1
        
        if count == 100:
            break
        
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

# Code starts here
    
# 1. Load Pretrained Model

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
# Load model with specific config
config = Qwen2VLConfig.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    config=config,
    torch_dtype=torch.float16,
    device_map="auto",
    use_flash_attention_2=False
)

# 2. Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="MULTI_MODAL_VISION_TEXT_GENERATION",
    inference_mode=False,
    bias="none"
)
model = get_peft_model(model, lora_config)

with open('/grphome/grp_handwriting/synthetic_data/lines/english_samples/OCR_9950000.json', 'r') as file:
    ImageTextJson = json.load(file)

dataset = create_dataset_from_json(ImageTextJson)

training_args = TrainingArguments(
    output_dir="/grphome/grp_handwriting/compute/qwen_finetune_bbox",  # Directory to save checkpoints
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_dir="/grphome/grp_handwriting/compute/qwen_finetune_test_bbox",  # Directory for logs
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    remove_unused_columns=False,
    fp16=True,
    dataloader_pin_memory=False
)

# 5. Define a Trainer
def collate_fn(batch):
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    bboxes = [item["bbox"] for item in batch]
    
    # Format text with image token and bbox information
    formatted_texts = [f"<|image_pad|> {text}" for text in texts]
    
    # Process using the processor
    batch_features = processor(
        images=images,
        text=formatted_texts,
        padding=True,
        return_tensors="pt"
    )
    
    # Use processor's config values
    patch_size = processor.image_processor.patch_size
    merge_size = processor.image_processor.merge_size
    
    # Calculate grid dimensions
    h = w = 224 // patch_size
    grid_thw = torch.tensor([[1, h, w]] * len(images), dtype=torch.long)
    
    # Create target sequences for bounding boxes
    target_texts = [f"Bounding box coordinates: {bbox}" for bbox in bboxes]
    target_features = processor(
        text=target_texts,
        padding=True,
        return_tensors="pt"
    )
    
    inputs = {
        "pixel_values": batch_features["pixel_values"],
        "input_ids": batch_features["input_ids"],
        "attention_mask": batch_features["attention_mask"],
        "image_grid_thw": grid_thw,
        "labels": target_features["input_ids"].clone(),
    }
    
    # Replace padding tokens with -100 in labels
    inputs["labels"][inputs["labels"] == processor.tokenizer.pad_token_id] = -100
    
    # Debug information
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"{k} shape:", v.shape)
            if k == "image_grid_thw":
                print(f"{k} values:", v)
            
    return inputs

# Add this before creating the trainer
print("\nProcessor config:")
print(f"Processor class: {processor.__class__.__name__}")
processor_config = processor.image_processor.to_dict()
print("Image processor config:", json.dumps(processor_config, indent=2))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    data_collator=collate_fn,
)

# 6. Fine-Tune
trainer.train()

# 7. Save Fine-Tuned Model
model.save_pretrained("/grphome/grp_handwriting/compute/qwen_finetune_bbox")
processor.save_pretrained("/grphome/grp_handwriting/compute/qwen_finetune_bbox")