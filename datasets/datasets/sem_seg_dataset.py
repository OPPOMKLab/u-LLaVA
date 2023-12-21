"""
Base dataset class of uLLaVA
Partially Adapted form: https://github.com/dvlab-research/LISA/blob/main/utils/sem_seg_dataset.py
"""

import os
import cv2
import copy
import torch
import numpy as np
from PIL import Image
from pycocotools import mask
import torch.nn.functional as F
from models.segment_anything.utils.transforms import ResizeLongestSide
from datasets.datasets.base_dataset import BaseDataset, preprocess, preprocess_image_text
from models import DEFAULT_SEG_TOKEN, DEFAULT_IMG_TOKEN, DEFAULT_TAG_START, DEFAULT_TAG_END


CLASS_TOKEN = '<class>'


class SemanticSegDataset(BaseDataset):
    class_map = {}

    def __init__(self,
                 vis_processor,
                 tokenizer,
                 vis_root,
                 ann_root,
                 template_root,
                 portion=1,
                 image_token_len=256,
                 seed=42,
                 data_type='image',
                 conv_type='conv_simple'
                 ):
        """
        vis_root (string):
        ann_root (string): jsonl file path
        split (string): val or test
        """
        super().__init__(
            vis_processor=vis_processor,
            tokenizer=tokenizer,
            vis_root=vis_root,
            ann_root=ann_root,
            template_root=template_root,
            seed=seed,
            portion=portion,
            data_type=data_type,
            conv_type=conv_type
        )

        self.image_token_len = image_token_len
        self.aspect_ratio = self.vis_processor.aspect_ratio

        self.num_sentence_per_item = 3
        # FOR SAM
        self.sam_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.sam_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.sam_size = 1024
        self.ignore_label = 255
        self.sam_transform = ResizeLongestSide(self.sam_size)

    def preprocess_for_sam(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.sam_mean) / self.sam_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.sam_size - h
        padw = self.sam_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    @staticmethod
    def get_label(label_path):
        label = Image.open(label_path)
        label = np.array(label)

        # ade20k
        label[label == 0] = 255
        label -= 1
        label[label == 254] = 255

        return label

    def build_sample(self, index):
        item = self.annotation[index]
        image_path = os.path.join(self.vis_root, item['image_path'])
        label_path = os.path.join(self.vis_root, item['label_path'])

        # single image, multi-round
        classes = item['classes']
        if len(classes) > self.num_sentence_per_item:
            sampled_classes = np.random.choice(classes, self.num_sentence_per_item, replace=False)
        else:
            sampled_classes = classes

        conversations = []
        roles = ['human', 'gpt']

        cls_seq = []
        for idx, cls in enumerate(sampled_classes):
            cls_name, cls_id = cls['class'], cls['class_id']
            question = self.random_choice_template().replace(CLASS_TOKEN, cls_name.lower())
            if idx != 0:
                question = question.replace(DEFAULT_IMG_TOKEN, '')

            conversations.append({
                'from': roles[0],
                'value': question,
            })
            conversations.append({
                'from': roles[1],
                'value': f'Sure. Info: {DEFAULT_SEG_TOKEN}; {DEFAULT_TAG_START}{cls_name.lower()}{DEFAULT_TAG_END}.'
            })

            cls_seq.append(cls_id)

        sample = {
            'image_path': image_path,
            'target': {
                'label_path': label_path,
                'class_sequence': cls_seq
            },
            'conversations': conversations
        }

        return sample

    def __getitem__(self, idx):
        sample = self.build_sample(idx)
        # image = Image.open(sample['image_path']).convert('RGB')
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        conversation_list = sample['conversations']
        label_path = sample["target"]["label_path"]
        cls_seq = sample["target"]["class_sequence"]

        label = self.get_label(label_path)

        # preprocess images for clip
        image_clip = self.vis_processor(image)
        image_sam = self.sam_transform.apply_image(image)  # preprocess images for sam
        resize = image_sam.shape[:2]

        # image_sam = self.preprocess_for_sam(pil_to_tensor(image_sam))
        image_sam = self.preprocess_for_sam(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())

        sources = preprocess_image_text(copy.deepcopy(conversation_list),
                                        cur_token_len=self.image_token_len)

        data_dict = preprocess(sources, self.tokenizer, self.conv_type)
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0])

        # For Segmentation
        seg_label = torch.from_numpy(label).long()
        masks = []
        for class_id in cls_seq:
            masks.append(seg_label == class_id)
        seg_mask = torch.stack(masks, dim=0).float()
        raw_size = [seg_mask.shape[1], seg_mask.shape[2]]

        image_dict = {'image': image_clip, 'image_sam': image_sam,
                      'seg_mask': seg_mask, 'raw_size': raw_size, 'resize': resize}

        data_dict.update(image_dict)

        return data_dict


class CocoStuffDataset(SemanticSegDataset):

    cocostuff_classes = []
    with open("./datasets/templates/cocostuff_classes.txt") as f:
        for line in f.readlines()[1:]:
            cocostuff_classes.append(line.strip().split(": ")[-1])
    cocostuff_classes = np.array(cocostuff_classes)

    class_map = {
        c: i for i, c in enumerate(cocostuff_classes)
    }

    def get_label(self, label_path):

        label = Image.open(label_path)
        label = np.array(label)

        for c, i in self.class_map.items():
            if "-" in c:
                label[label == i] = 255
        return label


class PacoDataset(SemanticSegDataset):
    def build_sample(self, index):
        item = self.annotation[index]
        image_path = os.path.join(self.vis_root, item['image_path'])
        anns = item['annotations']

        # 单图多轮
        classes = item['classes']
        indices = [_ for _ in range(len(classes))]
        if len(classes) > self.num_sentence_per_item:
            sampled_indices = np.random.choice(indices, self.num_sentence_per_item, replace=False)
            sampled_classes = [classes[i] for i in sampled_indices]
            sampled_anns = [anns[i] for i in sampled_indices]
        else:
            sampled_classes, sampled_anns = classes, anns

        conversations = []
        roles = ['human', 'gpt']
        for idx, cls in enumerate(sampled_classes):
            question = self.random_choice_template().replace(CLASS_TOKEN, cls.lower())
            if idx != 0:
                question = question.replace(DEFAULT_IMG_TOKEN, '')

            conversations.append({
                'from': roles[0],
                'value': question,
            })
            conversations.append({
                'from': roles[1],
                'value': f'Sure. Info: {DEFAULT_SEG_TOKEN}; {DEFAULT_TAG_START}{cls.lower()}{DEFAULT_TAG_END}.'
            })

        sample = {
            'image_path': image_path,
            'target': {
                'annotations': sampled_anns
            },
            'conversations': conversations
        }

        return sample

    def __getitem__(self, idx):
        sample = self.build_sample(idx)
        # image = Image.open(sample['image_path']).convert('RGB')
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        conversation_list = sample['conversations']
        sampled_anns = sample["target"]["annotations"]

        # preprocess images for clip
        image_clip = self.vis_processor(image)
        image_sam = self.sam_transform.apply_image(image)  # preprocess images for sam
        resize = image_sam.shape[:2]

        # image_sam = self.preprocess_for_sam(pil_to_tensor(image_sam))
        image_sam = self.preprocess_for_sam(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())

        sources = preprocess_image_text(copy.deepcopy(conversation_list),
                                        cur_token_len=self.image_token_len)

        data_dict = preprocess(sources, self.tokenizer, self.conv_type)
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0])

        # For Segmentation
        masks = []
        for ann in sampled_anns:
            height, width = ann['height'], ann['width']
            segmentation = ann['segmentation']

            if type(segmentation) == list:  # polygon
                rles = mask.frPyObjects(
                    segmentation, height, width
                )
                rle = mask.merge(rles)
            elif type(segmentation['counts']) == list:
                rle = mask.frPyObjects(segmentation, height, width)
            else:
                rle = ann['segmentation']
            m = mask.decode(rle)
            masks.append(m)

        masks = np.stack(masks, axis=0)
        seg_mask = torch.from_numpy(masks).float()
        # seg_label = torch.ones(seg_mask.shape[1], seg_mask.shape[2]) * self.ignore_label
        raw_size = [seg_mask.shape[1], seg_mask.shape[2]]

        image_dict = {'image': image_clip, 'image_sam': image_sam,
                      'seg_mask': seg_mask, 'raw_size': raw_size, 'resize': resize}

        data_dict.update(image_dict)

        return data_dict
