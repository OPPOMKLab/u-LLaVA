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

import os
import re
import sys
sys.path.append(os.getcwd())
import torch
import logging
import argparse
import transformers
from tqdm import tqdm
from PIL import Image
from peft import PeftModel
from tasks import setup_task
from torchvision.ops import box_iou
from metrics.bbox_iou import bbox_iou
from utils.config_builder import Config
from dataclasses import dataclass, field
from models.ullava import UllavaForCausalLM
from models.grounding_module import GroundingModule
from models import DEFAULT_TAG_START, DEFAULT_TAG_END
from torchvision.ops import masks_to_boxes, box_convert
from train_ullava import ModelArguments, TrainingArguments
from metrics.meter import AverageMeter, Summary, dict_to_cuda


@dataclass
class GroundingModelArguments(ModelArguments):
    grounding_path: str = field(default='./grounding_dino')


def select_bbox(dino_boxes, pred_mask, image_pil):
    w, h = image_pil.size
    dino_boxes = dino_boxes * torch.Tensor([w, h, w, h])
    dino_boxes = box_convert(boxes=dino_boxes, in_fmt="cxcywh", out_fmt="xyxy")

    # pred_mask > 0 after seg
    pred_mask = torch.tensor(pred_mask).unsqueeze(0)
    box_temp = masks_to_boxes(pred_mask)
    box_sam = box_temp[0]

    if len(dino_boxes) > 0:
        box = box_sam.expand(dino_boxes.size())
        ious = box_iou(box, dino_boxes)
        ious = torch.einsum('i i -> i', ious)  # take diag elem
        max_idx = torch.argmax(ious)
        return dino_boxes[max_idx], box_sam
    else:
        return box_sam, box_sam


def validate(model, grounding_module, tokenizer, val_dataset, data_collator):
    miou_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    mask2box_acc_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    mask2box_miou_meter = AverageMeter("Union", ":6.3f", Summary.SUM)

    model.eval()
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        collate_fn=data_collator
    )

    num = 0
    with tqdm(total=len(val_loader), file=sys.stdout) as pbar:
        for input_dict in val_loader:
            torch.cuda.empty_cache()
            input_dict['inference'] = True
            input_dict = dict_to_cuda(input_dict)

            with torch.no_grad():
                output_dict = model(**input_dict)

            logits = output_dict['logits']
            pred_masks = output_dict['pred_masks']
            output_ids = logits.argmax(-1)

            gt_box = torch.tensor(input_dict['bbox_list'][0])
            image_path = input_dict['image_list'][0]

            text_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            pattern = re.escape(DEFAULT_TAG_START) + '(.*?)' + re.escape(DEFAULT_TAG_END)
            tags = re.findall(pattern, text_output)
            grounding_tags = tags[-1] if len(tags) > 0 else ''

            image_pil = Image.open(image_path).convert('RGB')
            pred_boxes, logits, phrases = grounding_module.prompt2boxes(image_pil, grounding_tags)

            gt_box = box_convert(boxes=gt_box.unsqueeze(0), in_fmt="xywh", out_fmt="xyxy")[0]

            for i, pred_mask in enumerate(pred_masks):
                if pred_mask.shape[0] == 0 or len(pred_boxes) == 0:
                    continue

                pred_mask = pred_mask.detach().cpu().numpy()[0]
                pred_mask = pred_mask > 0

                try:
                    pred_box, box_sam = select_bbox(pred_boxes, pred_mask, image_pil)

                    m2b_res = bbox_iou(box_sam.unsqueeze(0), gt_box.unsqueeze(0))

                    mask2box_acc_meter.update(m2b_res['accuracy'] * 100.0, m2b_res['num'])
                    mask2box_miou_meter.update(m2b_res['miou'] * 100.0, m2b_res['num'])

                    res = bbox_iou(pred_box.unsqueeze(0), gt_box.unsqueeze(0))
                    miou_meter.update(res['miou'] * 100.0, res['num'])
                    acc_meter.update(res['accuracy'] * 100.0, res['num'])
                    num += 1

                except:
                    continue

            pbar.write('count {0} | Grounding: prec@0.5: {1:.2f}, mIoU: {2:.2f} | '
                       'Mask2box: prec@0.5: {3:.2f}, mIoU: {4:.2f}'.format(num, acc_meter.avg,
                                                                           miou_meter.avg, mask2box_acc_meter.avg,
                                                                           mask2box_miou_meter.avg))
            pbar.update(1)

    return acc_meter.avg, miou_meter.avg, mask2box_acc_meter.avg, mask2box_miou_meter.avg


def evaluate(config):
    os.makedirs(log_dir, exist_ok=True)
    model_args, dataset_args, eval_dataset_args, training_args, task_args, processor_args = config.assign_config()

    model_parser = transformers.HfArgumentParser(GroundingModelArguments)
    model_args = model_parser.parse_dict(model_args)[0]

    train_parser = transformers.HfArgumentParser(TrainingArguments)
    training_args = train_parser.parse_dict(training_args)[0]

    logger_name = model_args.llm_path.split('/')[-1] + "_grounding.log"
    logging.basicConfig(
        handlers=[logging.FileHandler(filename=os.path.join(log_dir, logger_name),
                                      encoding='utf-8-sig', mode='a+')],
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.llm_path,
        cache_dir=None,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        legacy=False
    )

    model = UllavaForCausalLM.from_pretrained(model_args.llm_path)

    if model_args.lora_r > 0:
        model.llm = PeftModel.from_pretrained(model.llm, model_args.llm_path)

    grounding_dino = GroundingModule(model_args.grounding_path, device='cuda')

    task = setup_task(task_args)
    processor_dict = task.build_processors(processor_args)
    data_collator = task.build_collator(tokenizer.pad_token_id)
    eval_dict = task.build_datasets(eval_dataset_args, tokenizer, processor_dict, conv_type=model_args.conv_type)

    model = model.to().cuda()

    model.eval()
    grounding_dino.eval()

    logger.info('build evaluation dataset done.')
    logger.info('Dataset|\tPrec@0.5\tmIoU\tPrec@0.5_m2b\tmIoU_m2b')

    for name, eval_dataset in eval_dict.items():
        acc, miou, acc_m2b, miou_m2b = validate(model, grounding_dino, tokenizer, eval_dataset, data_collator)
        logger.info('{0}|\t{1:.2f}|\t{2:.2f}|\t{3:.2f}|\t{4:.2f}'.format(name, acc, miou, acc_m2b, miou_m2b))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg_path", default='./configs/eval/eval_rec.yaml', help="path to configuration file.")
    parser.add_argument("--log_dir", default='./logs', help="path to save logs.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    log_dir = args.log_dir
    cfg = Config(args.cfg_path)
    cfg.pretty_print()
    evaluate(cfg)
