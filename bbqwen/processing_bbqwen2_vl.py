from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from typing import List, Union
import torch
from transformers.feature_extraction_sequence_utils import BatchFeature, ImageInput, VideoInput

class BBQwen2VLProcessor(Qwen2VLProcessor):
    """
    Processor for Qwen2-VL model with Bounding Box predictions.
    Inherits from Qwen2VLProcessor and adds handling for bbox_labels.
    """

    def __call__(
        self,
        images: Union[ImageInput, List[ImageInput]] = None,
        text: Union[str, List[str]] = None,
        videos: Union[VideoInput, List[VideoInput]] = None,
        bboxes: List[List[float]] = None,
        return_bbox_labels: bool = True,
        **kwargs,
    ) -> BatchFeature:
        features = super().__call__(images=images, text=text, videos=videos, **kwargs)

        if bboxes is not None and return_bbox_labels:
            # Convert bounding boxes to tensor and add to features
            bbox_tensor = torch.tensor(bboxes, dtype=torch.float32)
            features["bbox_labels"] = bbox_tensor

        return features
