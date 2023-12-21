"""
 Base processor of uLLaVA
 Adapted from: https://github.com/salesforce/LAVIS/blob/main/lavis/processors/alpro_processors.py
"""

from omegaconf import OmegaConf


class BaseProcessor:
    """
    Image and Text Preprocessor
    ImageProcessor: mean/std/size
    TextProcessor: prompt
    """
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, item):
        """
        :param item: PIL.Image(RGB) or Text
        :return:
        """
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        return cls()

    def build(self, **kwargs):
        cfg = OmegaConf.create(kwargs)

        return self.from_config(cfg)
