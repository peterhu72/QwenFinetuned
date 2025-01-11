from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig

class BBQwen2VLConfig(Qwen2VLConfig):
    """
    Configuration for Qwen2-VL model with Bounding Box predictions.
    Inherits from Qwen2VLConfig and adds bbox_size.
    """

    def __init__(
        self,
        *args,
        bbox_size: int = 4,  # Number of bounding box coordinates (x1, y1, x2, y2)
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bbox_size = bbox_size
        