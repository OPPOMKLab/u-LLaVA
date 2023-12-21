"""
Copyright 2023 OPPO

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import numpy as np
import torch.nn as nn
from yacs.config import CfgNode as CN
import models.GroundingDINO.groundingdino.datasets.transforms as T
from models.GroundingDINO.groundingdino.util.utils import clean_state_dict
from models.GroundingDINO.groundingdino.util.inference import predict, annotate
from models.GroundingDINO.groundingdino.models.GroundingDINO.groundingdino import build_groundingdino


def load_groundingdino_model(model_config_path, model_checkpoint_path):
    import gc
    args = CN.load_cfg(open(model_config_path, "r"))
    model = build_groundingdino(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print('loading GroundingDINO:', load_res)
    gc.collect()
    _ = model.eval()
    return model


class GroundingModule(nn.Module):
    def __init__(self, model_checkpoint, device='cpu'):
        super().__init__()
        groundingdino_config_file = "./configs/eval/GroundingDINO_SwinT_OGC.yaml"
        self.device = device

        self.grounding_model = load_groundingdino_model(groundingdino_config_file,
                                                        model_checkpoint).to(device)
        self.grounding_model.eval()

    @staticmethod
    def image_transform_grounding(image_pil):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image, _ = transform(image_pil, None)  # 3, h, w
        return image

    @torch.no_grad()
    def prompt2boxes(self, image_pil, prompt, box_threshold=0.25, text_threshold=0.2):
        prompt = prompt.lower()
        prompt = prompt.strip()
        if not prompt.endswith("."):
            prompt = prompt + "."

        image_tensor = self.image_transform_grounding(image_pil)

        # print('==> Box grounding with "{}"...'.format(prompt))

        boxes, logits, phrases = predict(self.grounding_model,
                                         image_tensor, prompt, box_threshold, text_threshold, device=self.device)

        return boxes, logits, phrases

    @torch.no_grad()
    def prompt2mask(self, image_pil, prompt, box_threshold=0.25, text_threshold=0.2):
        boxes, logits, phrases = self.prompt2boxes(image_pil, prompt, box_threshold, text_threshold)
        image_source = np.asarray(image_pil)
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        # cv2.imwrite("./annotated_image.jpg", annotated_frame)
        return annotated_frame

    @torch.no_grad()
    def annotate(self, image_source, boxes, logits, phrases, box_id=None):

        if box_id is not None:
            boxes, logits, phrases = boxes[box_id].unsqueeze(0), logits[box_id].unsqueeze(0), [phrases[box_id]]
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

        return annotated_frame
