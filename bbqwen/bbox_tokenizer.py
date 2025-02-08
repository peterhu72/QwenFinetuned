from typing import Dict, List, Union
import torch
from transformers import PreTrainedTokenizerBase
import numpy as np

class BBoxTokenizer:
    def __init__(self, base_tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = base_tokenizer
        print(f"Tokenizer type: {type(base_tokenizer)}")
        print(f"Has cls_token: {hasattr(base_tokenizer, 'cls_token_id')}")
        print(f"Has sep_token: {hasattr(base_tokenizer, 'sep_token_id')}")
        print(f"Has pad_token: {hasattr(base_tokenizer, 'pad_token_id')}")
        self.null_bbox = [0.0, 0.0, 0.0, 0.0]  # Null mask for padding tokens
        
    def normalize_bbox(self, bbox: List[Union[int, float]], image_width: int, image_height: int) -> List[float]:
        """Normalize bbox coordinates to 0-1 range."""
        return [
            float(bbox[0]) / image_width,
            float(bbox[1]) / image_height,
            float(bbox[2]) / image_width,
            float(bbox[3]) / image_height
        ]
    
    def __call__(self, 
                 text: str,
                 bbox: Union[List[int], List[float], List[List[Union[int, float]]]],
                 image_size: tuple,
                 padding: bool = True,
                 max_length: int = 512,
                 return_tensors: str = "pt") -> Dict:
        """
        Process text and corresponding bounding boxes.
        
        Args:
            text: Input text string
            bbox: Single bounding box [x1, y1, x2, y2] or list of boxes
            image_size: Tuple of (width, height) for normalization
            padding: Whether to pad sequences
            max_length: Maximum sequence length
            return_tensors: Output tensor type ("pt" for PyTorch)
        """
        if not text or not text.strip():
            raise ValueError(f"Empty or whitespace-only text received: '{text}'")
        
        print(f"Input text: {text}")
        print(f"Input bbox: {bbox}")
        
        # Ensure bbox is a list of lists
        if not isinstance(bbox[0], (list, tuple)):
            bbox = [bbox]  # Convert single bbox to list of bboxes
        print(f"Processed bbox: {bbox}")
        
        # Split text into words
        words = text.split() if isinstance(text, str) else text
        print(f"Split words: {words}")
        
        # If we have one bbox for the entire text, repeat it for each word
        if len(bbox) == 1 and len(words) > 1:
            bbox = bbox * len(words)
        print(f"Words count: {len(words)}, Bbox count: {len(bbox)}")
        
        # Ensure we have same number of words and bboxes
        if len(words) != len(bbox):
            raise ValueError(f"Number of words ({len(words)}) doesn't match number of bboxes ({len(bbox)})")
            
        # Normalize all bboxes
        try:
            normalized_boxes = [
                self.normalize_bbox(box, image_size[0], image_size[1])
                for box in bbox
            ]
            print(f"Normalized boxes: {normalized_boxes}")
        except (TypeError, IndexError) as e:
            raise ValueError(f"Invalid bbox format: {bbox}. Error: {e}")
        
        # Tokenize words and associate bboxes with subword tokens
        all_tokens = []
        all_bboxes = []
        
        for word, word_bbox in zip(words, normalized_boxes):
            word_tokens = self.tokenizer.tokenize(word)
            print(f"Word: {word}, Tokens: {word_tokens}")
            all_tokens.extend(word_tokens)
            all_bboxes.extend([word_bbox] * len(word_tokens))
        
        print(f"All tokens: {all_tokens}")
        
        # Convert tokens to ids
        input_ids = self.tokenizer.convert_tokens_to_ids(all_tokens)
        print(f"Input ids: {input_ids}")
        
        if not input_ids:
            raise ValueError(f"Failed to generate input_ids for text: {text}")
        
        # Handle special tokens and padding
        if padding:
            # Add [CLS] token at start
            if hasattr(self.tokenizer, 'cls_token_id') and self.tokenizer.cls_token_id is not None:
                input_ids = [self.tokenizer.cls_token_id] + input_ids
                all_bboxes = [self.null_bbox] + all_bboxes
            
            # Add [SEP] token at end
            if hasattr(self.tokenizer, 'sep_token_id') and self.tokenizer.sep_token_id is not None:
                input_ids = input_ids + [self.tokenizer.sep_token_id]
                all_bboxes = all_bboxes + [self.null_bbox]
            
            # Pad sequences
            if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                pad_length = max_length - len(input_ids)
                if pad_length > 0:
                    input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
                    all_bboxes = all_bboxes + [self.null_bbox] * pad_length
            else:
                print("Warning: Tokenizer does not have pad_token_id")
                pad_length = 0
            
            attention_mask = [1] * (len(all_bboxes) - pad_length) + [0] * pad_length
        else:
            attention_mask = [1] * len(all_bboxes)
        
        print(f"Final input_ids length: {len(input_ids)}")
        print(f"Final attention_mask length: {len(attention_mask)}")
        print(f"Final bbox_tensors length: {len(all_bboxes)}")
        
        # Ensure all tensors have values
        if not input_ids or not attention_mask or not all_bboxes:
            raise ValueError(
                f"Empty tensors detected: input_ids({len(input_ids)}), "
                f"attention_mask({len(attention_mask)}), "
                f"bbox_tensors({len(all_bboxes)})"
            )
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            try:
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "bbox_tensors": torch.tensor(all_bboxes, dtype=torch.float)
                }
            except Exception as e:
                print(f"Error creating tensors: {e}")
                print(f"input_ids: {input_ids}")
                print(f"attention_mask: {attention_mask}")
                print(f"all_bboxes: {all_bboxes}")
                raise
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bbox_tensors": all_bboxes
        }