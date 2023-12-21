from utils.registry import registry
from datasets.processors.clip_processor import CLIPProcessor
from datasets.processors.base_processor import BaseProcessor
from datasets.processors.video_processor import VideoTrainProcessor, VideoEvalProcessor, GIFTrainProcessor

__all__ = [
    "BaseProcessor",
    "CLIPProcessor",
    "VideoTrainProcessor",
    "GIFTrainProcessor"
]


def load_processor(name, cfg=None):
    """
    Example

    processor = load_processor("video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
