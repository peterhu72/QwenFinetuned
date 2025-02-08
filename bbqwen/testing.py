import json
from PIL import Image


def create_dataset_from_json(json_data):
        dataset = []
        done = False  # Flag to control breaking from nested loops

        for filename, info in json_data.items():
            if done:
                break
            
            sections = info["sections"]  # Extract the text as the label
            image_path = f"/home/tarch/datasets/synthetic/lines/english/{filename}.jpg"  # Construct the full path
        
            for section in sections:
                if done:
                    break
                try:
                    # Load and convert the image to RGB format
                    image = Image.open(image_path).convert("RGB")
                    #processed_image = transform_image(image)
                
                except FileNotFoundError:
                    print(f"Warning: {filename} not found at {image_path}")
                    continue
                
                for paragraph in section["paragraphs"]:
                    if done:
                        break
                    
                    for line in paragraph["lines"]:
                        if len(dataset) >= 10:
                            done = True
                            break
                            
                        dataset.append({
                            "text": line["text"],
                            "image": image,
                            "bbox": line["bbox"]
                        })
                        
                        for word in line["words"]:
                            if len(dataset) >= 10:
                                done = True
                                break
                                
                            dataset.append({
                                "text": word["text"],
                                "image": image,
                                "bbox": word["bbox"]
                            })
                        if done:
                            break

        return dataset
    
    
    
data = "/grphome/grp_handwriting/synthetic_data/lines/english_samples/OCR_9950000.json"

with open(data, "r") as f:
    data = json.load(f)

dataset = create_dataset_from_json(data)

print(dataset[0]["bbox"][0])