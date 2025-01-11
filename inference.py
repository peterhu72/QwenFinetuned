from easydict import EasyDict as edict

from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor
from bbqwen.modeling_bbqwen2_vl import BBQwen2VLForConditionalGeneration
from bbqwen.configuration_bbqwen2_vl import BBQwen2VLConfig
import time
from functools import wraps
import traceback

def load_model(model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Load model with optimizations."""
    config = BBQwen2VLConfig.from_pretrained(model_path)
    
    model = BBQwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load processor and check if it's a bbox-enabled checkpoint
    processor = AutoProcessor.from_pretrained(model_path)
    
    return model, processor

def prepare_image(image_path: Path):
    """Load and prepare image."""
    return Image.open(image_path)
    

@torch.inference_mode()
def generate_with_bbox(
    model, 
    processor, 
    image: Image.Image,
    prompt: str = "Transcribe this image.",
    max_new_tokens: int = 128
):
    """Generate text and bounding boxes for an image."""
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                # {
                #     "type": "text",
                #     "text": prompt
                # }
            ]
        }
    ]
    
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    
    inputs = edict({k: v.to(model.device) for k, v in inputs.items()})
    
    # First get the model outputs directly to get bbox predictions
    model_outputs = model(**inputs)
    bbox_predictions = model_outputs.bbox_logits.sigmoid().cpu().numpy()
    
    # Then generate the text
    output_ids = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,
    )
    
    generated_ids = output_ids[0][len(inputs.input_ids[0]):]
    generated_text = processor.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    return generated_text, bbox_predictions

    
def retry_on_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print("\nError occurred:")
                print(traceback.format_exc())  # Print full stack trace
                
                while True:
                    choice = input("\nWhat would you like to do?\n[R]etry, [S]kip, or [E]xit: ").lower()
                    if choice in ['r', 's', 'e']:
                        break
                    print("Invalid choice. Please try again.")
                
                if choice == 'r':
                    continue
                elif choice == 's':
                    return None
                else:  # 'e'
                    raise e
    return wrapper

@retry_on_error
def process_single_image(image_path: Path, model, processor):
    """Process a single image and return generated text."""
    image = prepare_image(image_path)
    generated_text, bbox = generate_with_bbox(model, processor, image)
    print(f"bbox: {bbox}")
    return generated_text

def main():
    model_path = "Qwen/Qwen2-VL-2B-Instruct"
    initial_image_path = Path("F:/data/standard_hwr/IAM_OFFLINE/lines/a01/a01-000u/a01-000u-00.png")
    
    # Load model and processor
    print("Loading model...")
    model, processor = load_model(model_path)
    
    while True:
        if 'image_path' not in locals():
            image_path = initial_image_path
        else:
            # Get new path from user
            user_input = input("\nEnter image path (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            image_path = Path(user_input)
        
        result = process_single_image(image_path, model, processor)
        if result is not None:
            print(f"\nResults for {image_path}:")
            print(f"Generated Text: {result}")


if __name__ == "__main__":
    main() 