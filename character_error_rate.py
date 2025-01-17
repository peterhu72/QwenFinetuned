import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import json
from torch.utils.data import DataLoader

def create_dataset_from_json(json_data):
    dataset = []
    count = 0

    for filename, info in json_data.items():
        if len(dataset) >= 50000:
            break
        sections = info["sections"]
        image_path = f"/home/tarch/datasets/synthetic/lines/english/{filename}.jpg"
        
        try:
            image = Image.open(image_path).convert("RGB")
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
                        "image": image
                    })

    print(f"Created dataset with {len(dataset)} samples")
    return dataset

def collate_fn(batch):
    # Modified to handle single items
    return {
        "text": [item["text"] for item in batch],
        "images": [item["image"] for item in batch]
    }

def calculate_cer(true_texts, predicted_texts):
    def levenshtein_distance(ref, hyp):
        n, m = len(ref), len(hyp)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            for j in range(m + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif ref[i - 1] == hyp[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
        return dp[n][m]

    total_distance = sum(levenshtein_distance(true, pred) for true, pred in zip(true_texts, predicted_texts))
    total_chars = sum(len(text) for text in true_texts)
    cer = total_distance / total_chars if total_chars > 0 else 0.0
    return cer

def process_single_message(processor, model, image):
    # Create a single message for processing
    message = {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Output the text in this image."}, 
        ],
    }
    
    text_input = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    images, video_inputs = process_vision_info([message])
    
    inputs = processor(text=[text_input], images=images, videos=video_inputs, padding=True, return_tensors="pt").to(DEVICE)
    
    # Generate prediction
    outputs = model.generate(**inputs, max_new_tokens=128)
    outputs_trimmed = outputs[:, len(inputs.input_ids[0]):]
    prediction = processor.batch_decode(outputs_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    # Clean up the prediction
    prediction = prediction.replace("The text in the image is: ", "")
    prediction = prediction.replace("The text is: ", "")
    prediction = prediction.replace("Text: ", "")
    prediction = prediction.replace("Output: ", "")
    prediction = prediction.replace("The image shows: ", "")
    prediction = prediction.replace("The image contains: ", "")
    prediction = prediction.strip()
    
    return prediction

def evaluate_model_on_cer(processor, model, dataloader, desc=""):
    model.eval()
    all_pred_texts = []
    all_true_texts = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {desc}"):
            # Process each image individually
            for img, true_text in zip(batch["images"], batch["text"]):
                pred_text = process_single_message(processor, model, img)
                all_pred_texts.append(pred_text)
                all_true_texts.append(true_text)
                print(f"True text: {true_text}")
                print(f"Predicted text: {pred_text}\n")

    cer = calculate_cer(all_true_texts, all_pred_texts)
    return cer

# Use GPU if available
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load models and processors
print("Loading Finetuned Model...")
processor1 = AutoProcessor.from_pretrained("/grphome/grp_handwriting/compute/qwen_finetune_test2")
model1 = Qwen2VLForConditionalGeneration.from_pretrained(
    "/grphome/grp_handwriting/compute/qwen_finetune_test2", 
    torch_dtype=torch.bfloat16
).to(DEVICE)

print("Loading Base Model...")
processor2 = AutoProcessor.from_pretrained("/grphome/grp_handwriting/compute/qwen2_vl_2b_instruct")
model2 = Qwen2VLForConditionalGeneration.from_pretrained(
    "/grphome/grp_handwriting/compute/qwen2_vl_2b_instruct", 
    torch_dtype=torch.bfloat16
).to(DEVICE)

# Load dataset
print("Loading dataset...")
with open('/grphome/grp_handwriting/synthetic_data/lines/english_samples/OCR_9950000.json', 'r') as file:
    test_data = json.load(file)

# Create and split dataset
dataset = create_dataset_from_json(test_data)
first_100k_data = dataset[:5000]  # Training set
second_100k_data = dataset[25000:30000]  # Test set

# Create DataLoaders - batch_size=1 for single message processing
first_loader = DataLoader(first_100k_data, batch_size=1, collate_fn=collate_fn)
second_loader = DataLoader(second_100k_data, batch_size=1, collate_fn=collate_fn)

# Evaluate both models on both sets
results = {}

print("\nEvaluating Finetuned Model...")
print('Train')
results['finetuned_train'] = evaluate_model_on_cer(processor1, model1, first_loader, "finetuned model on training set")
print('Test')
results['finetuned_test'] = evaluate_model_on_cer(processor1, model1, second_loader, "finetuned model on test set")

print("\nEvaluating Base Model...")
print('Train')
results['base_train'] = evaluate_model_on_cer(processor2, model2, first_loader, "base model on training set")
print('Test')
results['base_test'] = evaluate_model_on_cer(processor2, model2, second_loader, "base model on test set")

# Print final results
print("\nFinal Results:")
print(f"Finetuned Model - Training CER: {results['finetuned_train']:.4f}")
print(f"Finetuned Model - Test CER: {results['finetuned_test']:.4f}")
print(f"Base Model - Training CER: {results['base_train']:.4f}")
print(f"Base Model - Test CER: {results['base_test']:.4f}")