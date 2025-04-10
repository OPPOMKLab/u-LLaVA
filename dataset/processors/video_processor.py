"""
 video and gif processors of uLLaVA
 Adapted from: https://github.com/DAMO-NLP-SG/Video-LLaMA/blob/main/video_llama/processors/base_processor.py
"""
import torch
import decord
import numpy as np
import random as rnd
import imageio.v3 as iio
from decord import VideoReader
from omegaconf import OmegaConf
from torchvision import transforms
from utils.registry import registry
from dataset.tools import transforms_video
from dataset.tools import functional_video as F
from dataset.processors.base_processor import BaseProcessor


MAX_INT = registry.get("MAX_INT")
decord.bridge.set_bridge("torch")


class VideoBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None, n_frm=MAX_INT):
        super(VideoBaseProcessor, self).__init__()
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms_video.NormalizeVideo(mean, std)

        self.n_frm = n_frm

    @staticmethod
    def load_video(video_path, n_frm=MAX_INT, height=-1, width=-1, sampling="uniform", return_msg=False):
        decord.bridge.set_bridge("torch")
        vr = VideoReader(uri=video_path, height=height, width=width)

        vlen = len(vr)
        start, end = 0, vlen
        n_frm = min(n_frm, vlen)

        if sampling == "uniform":
            indices = np.arange(start, end, vlen / n_frm).astype(int).tolist()
        elif sampling == "headtail":
            indices_h = sorted(rnd.sample(range(vlen // 2), n_frm // 2))
            indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frm // 2))
            indices = indices_h + indices_t
        else:
            raise NotImplementedError

        # get_batch -> T, H, W, C
        temp_frms = vr.get_batch(indices)
        # print(type(temp_frms))
        tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
        frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

        if not return_msg:
            return frms

        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
        return frms, msg

    @staticmethod
    def load_gif(gif_path, n_frm=MAX_INT, sampling="uniform", return_msg=False):
        """
        使用imageio读取gif, 返回的为np.array, [n_frm, H, W, C], 长度为帧数, RGB, HWC
        :param gif_path:
        :param n_frm:
        :param sampling:
        :param return_msg:
        :return:
        """
        # index=None means: read all images in the file and stack along first axis
        frames = iio.imread(gif_path, pilmode='RGB', index=None)
        glen = frames.shape[0]
        start, end = 0, glen
        n_frm = min(n_frm, glen)

        if sampling == "uniform":
            indices = np.arange(start, end, glen / n_frm).astype(int).tolist()
        elif sampling == "headtail":
            indices_h = sorted(rnd.sample(range(glen // 2), n_frm // 2))
            indices_t = sorted(rnd.sample(range(glen // 2, glen), n_frm // 2))
            indices = indices_h + indices_t
        else:
            raise NotImplementedError

        # get_batch -> T, H, W, C

        tensor_frms = torch.from_numpy(frames[indices])
        # (C, T, H, W)
        frms = tensor_frms.permute(3, 0, 1, 2).float()

        if not return_msg:
            return frms

        # " " should be added in the start and end
        msg = f"The video contains {len(indices)} frames sampled. "
        return frms, msg


class ToUint8(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.to(torch.uint8)

    def __repr__(self):
        return self.__class__.__name__


class ToTHWC(object):
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (C, T, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, H, W, C)
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.permute(1, 2, 3, 0)

    def __repr__(self):
        return self.__class__.__name__


class MeanVideo(object):
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (C, T, H, W)
    Return:
        mean the video clip in temporal sequence
        clip (torch.tensor, dtype=torch.float): Size is (C, H, W)
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.mean(dim=1).squeeze()

    def __repr__(self):
        return self.__class__.__name__


class ResizeVideo(object):
    def __init__(self, target_size, interpolation_mode="bilinear"):
        self.target_size = target_size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, crop_size, crop_size)
        """
        return F.resize(clip, self.target_size, self.interpolation_mode)

    def __repr__(self):
        return self.__class__.__name__ + "(resize_size={0})".format(self.target_size)


@registry.register_processor("video_train")
class VideoTrainProcessor(VideoBaseProcessor):
    def __init__(
        self,
        image_size=384,
        mean=None,
        std=None,
        min_scale=0.5,
        max_scale=1.0,
        n_frm=MAX_INT,
    ):
        super().__init__(mean=mean, std=std, n_frm=n_frm)

        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                # Video size is (C, T, H, W)
                transforms_video.RandomResizedCropVideo(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation_mode="bicubic",
                ),
                ToTHWC(),  # C, T, H, W -> T, H, W, C
                ToUint8(),
                transforms_video.ToTensorVideo(),  # T, H, W, C -> C, T, H, W
                self.normalize,
                # MeanVideo(),  # C, T, H, W -> C, H, W
            ]
        )

    def __call__(self, vpath):
        """
        Args:
            vpath: Video path that Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (T, C, size, size).
        """
        clip = self.load_video(
            video_path=vpath,
            n_frm=self.n_frm,
            height=self.image_size,
            width=self.image_size,
            sampling="headtail",
        )

        return self.transform(clip)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        n_frm = cfg.get("n_frm", MAX_INT)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
            n_frm=n_frm,
        )


@registry.register_processor("video_eval")
class VideoEvalProcessor(VideoBaseProcessor):
    def __init__(self, image_size=256, mean=None, std=None, n_frm=MAX_INT):
        super().__init__(mean=mean, std=std, n_frm=n_frm)

        self.image_size = image_size

        # Input video size is (C, T, H, W)
        self.transform = transforms.Compose(
            [
                # frames will be resized during decord loading.
                ToUint8(),  # C, T, H, W
                ToTHWC(),  # T, H, W, C
                transforms_video.ToTensorVideo(),  # C, T, H, W
                self.normalize,  # C, T, H, W
                # MeanVideo(),  # C, T, H, W -> C, H, W
            ]
        )

    def __call__(self, vpath):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (T, C, size, size).
        """
        clip = self.load_video(
            video_path=vpath,
            n_frm=self.n_frm,
            height=self.image_size,
            width=self.image_size,
        )

        return self.transform(clip)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        n_frm = cfg.get("n_frm", MAX_INT)

        return cls(image_size=image_size, mean=mean, std=std, n_frm=n_frm)


@registry.register_processor("gif_train")
class GIFTrainProcessor(VideoBaseProcessor):
    def __init__(
        self,
        image_size=384,
        mean=None,
        std=None,
        min_scale=0.5,
        max_scale=1.0,
        n_frm=MAX_INT,
    ):
        super().__init__(mean=mean, std=std, n_frm=n_frm)

        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                # Video size is (C, T, H, W)
                transforms_video.RandomResizedCropVideo(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation_mode="bicubic",
                ),
                ToTHWC(),  # C, T, H, W -> T, H, W, C
                ToUint8(),
                transforms_video.ToTensorVideo(),  # T, H, W, C -> C, T, H, W
                self.normalize,
                # MeanVideo(),  # C, T, H, W -> C, H, W
            ]
        )

    def __call__(self, gpath):
        """
        Args:
            gpath: GIF path that Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (T, C, size, size).
        """
        clip = self.load_gif(
            gif_path=gpath,
            n_frm=self.n_frm,
            sampling="headtail",
        )

        return self.transform(clip)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        n_frm = cfg.get("n_frm", MAX_INT)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
            n_frm=n_frm,
        )
