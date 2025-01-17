import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

'''
This File is for aplying inference on Qwen models
'''


DEVICE = "cuda:0"
# default: Load the model on the available device(s)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-2B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("/grphome/grp_handwriting/compute/qwen_finetune_test2")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/grphome/grp_handwriting/compute/qwen_finetune_test2", torch_dtype=torch.bfloat16
).to(DEVICE)
# processor = AutoProcessor.from_pretrained("/grphome/grp_handwriting/compute/qwen2_vl_2b_instruct")
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "/grphome/grp_handwriting/compute/qwen2_vl_2b_instruct", torch_dtype=torch.bfloat16
# ).to(DEVICE)

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/home/tarch/datasets/synthetic/lines/english/009900009.jpg",
            },
            {"type": "text", "text": "Only output the handwritten text in this image and nothing else"},
        ],
    }
]

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "/home/tarch/datasets/synthetic/lines/english_samples/009949003.jpg",
#             },
#             {"type": "text", "text": "Give two pixel coordinates denoting a bounding box around the word 'improve'. The first coordinate is the top left corner of the box and the second coordinate is the bottom right corner of the box. This image is 1152 x 64 px."},
#         ],
#     }
# ]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)