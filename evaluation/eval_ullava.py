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
import sys
sys.path.append(os.getcwd())
import torch
import logging
import argparse
import transformers
from tqdm import tqdm
from peft import PeftModel
from tasks import setup_task
from utils.config_builder import Config
from models.ullava import UllavaForCausalLM
from train_ullava import ModelArguments, TrainingArguments
from evaluation.tools import intersectionAndUnionGPU, AverageMeter, Summary, dict_to_cuda, bbox_iou


def validate(model, val_dataset, data_collator, dtype):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    prec05_meter = AverageMeter("Prec@0.5", ":6.3f", Summary.SUM)

    model.eval()
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        collate_fn=data_collator
    )

    for input_dict in tqdm(val_loader):
        torch.cuda.empty_cache()
        input_dict['inference'] = True

        input_dict = dict_to_cuda(input_dict, dtype)

        with torch.no_grad():
            output_dict = model(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()

        pred_boxes = output_dict["pred_boxes"][0]
        boxes_list = output_dict["gt_boxes"][0]

        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            try:
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )

                intersection_i, union_i = intersection_i.cpu().numpy(), union_i.cpu().numpy()

                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0  # no-object target
            except:
                continue
        acc_iou = acc_iou / masks_list.shape[0]
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

        for pred_box, gt_box in zip(pred_boxes, boxes_list):
            res = bbox_iou(pred_box.unsqueeze(0), gt_box.unsqueeze(0))
            prec05_meter.update(res['accuracy'] * 100.0, res['num'])

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1] * 100.0
    giou = acc_iou_meter.avg[1] * 100.0
    prec05 = prec05_meter.avg

    print("ciou: {:.2f}, giou: {:.2f}, prec@0.5: {:.2f} success: {}".format(ciou,
                                                                            giou,
                                                                            prec05,
                                                                            intersection_meter.count))

    return ciou, giou, prec05


def evaluate(config):
    os.makedirs(log_dir, exist_ok=True)
    model_args, dataset_args, eval_dataset_args, training_args, task_args, processor_args = config.assign_config()

    logger_name = model_args.llm_path.split('/')[-1] + ".log"
    logging.basicConfig(
        handlers=[logging.FileHandler(filename=os.path.join(log_dir, logger_name),
                                      encoding='utf-8-sig', mode='a+')],
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger()

    model_parser = transformers.HfArgumentParser(ModelArguments)
    model_args = model_parser.parse_dict(model_args)[0]

    train_parser = transformers.HfArgumentParser(TrainingArguments)
    training_args = train_parser.parse_dict(training_args)[0]

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.llm_path,
        cache_dir=None,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        legacy=False
    )
    dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16 if args.dtype == 'bf16' else torch.float32
    model = UllavaForCausalLM.from_pretrained(model_args.llm_path, torch_dtype=dtype)

    if model_args.lora_r > 0:
        model.llm = PeftModel.from_pretrained(model.llm, model_args.llm_path, torch_type=dtype)

    task = setup_task(task_args)
    processor_dict = task.build_processors(processor_args)
    data_collator = task.build_collator(tokenizer.pad_token_id)

    model.cuda()
    model.eval()

    eval_dict = task.build_datasets(eval_dataset_args, tokenizer, processor_dict, conv_type=model_args.conv_type)
    logger.info('build evaluation dataset done.')
    logger.info('Dataset|\tcIoU\tgIoU')

    for name, eval_dataset in eval_dict.items():
        ciou, giou, prec05 = validate(model, val_dataset=eval_dataset, data_collator=data_collator, dtype=dtype)
        logger.info('{0}|\t{1:.2f}|\t{2:.2f}\t{3:.3f}'.format(name, ciou, giou, prec05))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg_path", default='./configs/eval/eval_salient.yaml', help="path to configuration file.")
    parser.add_argument("--log_dir", default='./logs', help="path to save logs.")
    parser.add_argument(
        "--dtype",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
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
