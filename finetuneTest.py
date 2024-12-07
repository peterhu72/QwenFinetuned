import torch
import json
from PIL import Image
from torch.utils.data import DataLoader
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLConfig,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from torchvision import transforms
from collections import defaultdict

'''

This File is for finetuning Qwen in order to predict what an image says. 
Important to note:
We are resizing the images to 224 x 224 for finetuning
etc...

'''

# Create a more robust image transformation
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
        if count == 100000:
            break
        
        sections = info["sections"]
        image_path = f"/home/tarch/datasets/synthetic/lines/english/{filename}.jpg"
        
        try:
            image = Image.open(image_path).convert("RGB")
            processed_image = transform_image(image)
            
            # Verify image tensor shape
            assert processed_image.shape == (3, 224, 224), f"Unexpected image shape: {processed_image.shape}"
            
        except FileNotFoundError:
            print(f"Warning: {filename} not found at {image_path}")
            continue
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
        
        for section in sections:
            for paragraph in section["paragraphs"]:
                for line in paragraph["lines"]:
                    dataset.append({
                        "text": line["text"],
                        "image": processed_image
                    })

    print(f"Created dataset with {len(dataset)} samples")
    return dataset

def collate_fn(batch):
    images = [transforms.ToPILImage()(item["image"]) for item in batch]
    texts = [f"<|image_pad|> {item['text']}" for item in batch]
    
    # Process using the processor
    batch_features = processor(
        images=images,
        text=texts,
        padding=True,
        return_tensors="pt"
    )
    
    # Use processor's config values
    patch_size = processor.image_processor.patch_size  # Should be 14
    merge_size = processor.image_processor.merge_size  # Should be 2
    
    print('CHECK', patch_size)
    print('CHECK', merge_size)
    
    # Calculate grid dimensions
    h = w = 224 // patch_size
    grid_thw = torch.tensor([[1, h, w]] * len(images), dtype=torch.long)
    
    inputs = {
        "pixel_values": batch_features["pixel_values"],
        "input_ids": batch_features["input_ids"],
        "attention_mask": batch_features["attention_mask"],
        "image_grid_thw": grid_thw,
        "labels": batch_features["input_ids"].clone(),
    }
    
    inputs["labels"][inputs["labels"] == processor.tokenizer.pad_token_id] = -100
    
    # Debug prints
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"{k} shape:", v.shape)
            if k == "image_grid_thw":
                print(f"{k} values:", v)

    return inputs


# Code starts here

# Load the processor and model
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
# Load model with specific config
config = Qwen2VLConfig.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Load model with config
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    config=config,
    torch_dtype=torch.float16,
    device_map="auto",
    use_flash_attention_2=False
)

# Apply LoRA
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

# Load your dataset
with open('/grphome/grp_handwriting/synthetic_data/lines/english_samples/OCR_9950000.json', 'r') as file:
    ImageTextJson = json.load(file)

dataset = create_dataset_from_json(ImageTextJson)

# Prepare trainer
training_args = TrainingArguments(
    output_dir="/grphome/grp_handwriting/compute/qwen_finetune_test",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_dir="/grphome/grp_handwriting/compute/qwen_finetune_test_logs",
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    remove_unused_columns=False,
    fp16=True,
    dataloader_pin_memory=False
)

# Add this before creating the trainer
print("\nProcessor config:")
print(f"Processor class: {processor.__class__.__name__}")
processor_config = processor.image_processor.to_dict()
print("Image processor config:", json.dumps(processor_config, indent=2))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

# Fine-tune
trainer.train()

# 7. Save Fine-Tuned Model
model.save_pretrained("/grphome/grp_handwriting/compute/qwen_finetune_test")
processor.save_pretrained("/grphome/grp_handwriting/compute/qwen_finetune_test")