"""
Base dataset class of uLLaVA
Partially Adapted form: https://github.com/haotian-liu/LLaVA/blob/main/llava/train/train.py
"""

import json
import copy
import random
import pathlib
import numpy as np
import transformers
from typing import Sequence, Dict
from torch.utils.data import Dataset
from utils.conversation import conversation_lib, SeparatorStyle
from models import DEFAULT_IMG_PATCH_TOKEN, DEFAULT_IMG_START_TOKEN, DEFAULT_IMG_TOKEN, \
    DEFAULT_IMG_END_TOKEN, DEFAULT_VID_PATCH_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, IGNORE_INDEX


class BaseDataset(Dataset):
    def __init__(
            self,
            vis_processor=None,
            tokenizer=None,
            vis_root='',
            ann_root='',
            template_root='',
            portion=1,
            seed=42,
            data_type='image',
            conv_type='conv_simple'
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.seed = seed

        # necessary
        self.annotation = self.get_annotations(ann_root, portion)
        self.tokenizer = tokenizer

        # Optional
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.templates = self.get_templates(template_root) if template_root != '' else None

        self.rng = np.random.default_rng(self.seed)

        self.data_type = data_type
        self.conv_type = conv_type

        # self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, item):
        pass

    def get_annotations(self, ann_root, portion):
        if ann_root.endswith('json'):
            data_path = pathlib.Path(ann_root)
            with data_path.open(encoding='utf-8') as f:
                annotation = json.load(f)
        elif ann_root.endswith('jsonl'):
            data_path = pathlib.Path(ann_root)
            annotation = []
            with open(data_path, 'r', encoding='utf8') as f:
                for line in f:
                    # convert str to dict
                    line = json.loads(line)
                    annotation.append(line)
        else:
            print('Annotation file should be json or jsonl format!')
            raise NotImplementedError

        if portion < 1.0:
            n_annotation = len(annotation)
            n_sampled = int(n_annotation * portion)

            # set to the same seed to get the same subset when ddp
            random.seed(self.seed)
            annotation = random.sample(annotation, n_sampled)

        return annotation

    @staticmethod
    def get_templates(template_root):
        assert template_root.endswith('json')
        template_path = pathlib.Path(template_root)
        templates = json.load(open(template_path, 'r', encoding='utf8'))
        return templates

    def template_nums(self):
        return len(self.templates)

    def random_choice_template(self):
        return self.rng.choice(self.templates)

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)


def _add_speaker_and_signal(header, source, roles, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "###"
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = roles[0]
        elif from_str.lower() == "gpt":
            from_str = roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer,
                 truncation=True) -> Dict:
    """
    Tokenize a list of strings.
    :param strings:
    :param tokenizer:
    :param truncation: True to save memory
    :return:
    """
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=truncation,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def preprocess_sep1(
        sources: Sequence[dict],
        tokenizer: transformers.PreTrainedTokenizer,
        conv_type: str
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    header = ''
    conversations = []
    conv = conversation_lib[conv_type].copy()
    for source in sources:
        header = f"{conv.system}\n\n"
        conversation = _add_speaker_and_signal(header, source, conv.roles)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def preprocess_sep2(
        sources: Sequence[dict],
        tokenizer: transformers.PreTrainedTokenizer,
        conv_type: str
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    For Sep 2 type conversation
    """
    # add end signal and concatenate together
    conv = conversation_lib[conv_type].copy()
    assert conv.sep_style == SeparatorStyle.TWO or conv.sep_style == SeparatorStyle.LLAMA_2
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)

    if conv_type == 'conv_llama2':
        sep = "[/INST] "
    else:
        sep = conv.sep + conv.roles[1] + ": "

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids)
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
                
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(input_ids=input_ids, labels=targets)


def preprocess_image_text(
        conversation_list: Sequence[dict],
        cur_token_len: int,
) -> Sequence:
    # Image Token process
    image_token_len = cur_token_len
    for sentence in conversation_list:
        # put image token fisrt
        if DEFAULT_IMG_TOKEN in sentence['value']:
            sentence['value'] = sentence['value'].replace(DEFAULT_IMG_TOKEN, '').strip()
            sentence['value'] = DEFAULT_IMG_TOKEN + '\n' + sentence['value']
            sentence['value'] = sentence['value'].strip()

        replace_token = DEFAULT_IMG_PATCH_TOKEN * image_token_len
        replace_token = DEFAULT_IMG_START_TOKEN + replace_token + DEFAULT_IMG_END_TOKEN
        sentence["value"] = sentence["value"].replace(DEFAULT_IMG_TOKEN, replace_token)

    return [conversation_list]


def preprocess_video_text(
        conversation_list: Sequence[dict],
        cur_token_len: int,
) -> Sequence:
    # Video\GIF
    replace_token = DEFAULT_VID_START_TOKEN + DEFAULT_VID_PATCH_TOKEN * cur_token_len + DEFAULT_VID_END_TOKEN
    conversation_list[0]["value"] += replace_token

    return [conversation_list]


def preprocess(
        sources: Sequence[dict],
        tokenizer: transformers.PreTrainedTokenizer,
        conv_type: str
) -> Dict:
    if conv_type == 'conv_simple':
        return preprocess_sep1(sources, tokenizer, conv_type)
    elif conv_type == 'conv_sep2' or conv_type == 'conv_llama2':
        return preprocess_sep2(sources, tokenizer, conv_type)
    else:
        raise NotImplementedError
