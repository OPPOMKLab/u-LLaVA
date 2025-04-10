from utils.registry import registry
from dataset.processors.clip_processor import CLIPProcessor
from dataset.processors.base_processor import BaseProcessor
from dataset.processors.video_processor import VideoTrainProcessor, VideoEvalProcessor, GIFTrainProcessor

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
