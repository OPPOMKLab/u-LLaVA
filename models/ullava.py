"""
u-LLaVA with segmentation module
SAM patch is Adapted form: https://github.com/dvlab-research/LISA/blob/main/model/LISA.py
"""

import copy
import torch
import torch.nn as nn
from typing import List
from utils.registry import registry
from models.segment_anything import build_sam_vit_h
from models.ullava_core import UllavaCoreConfig, UllavaCoreForCausalLM
from models.loss import bbox_l1_loss, bbox_giou_loss, sigmoid_ce_loss, dice_loss
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig, AutoConfig


class UllavaConfig(PretrainedConfig):
    model_type = "ullava"
    is_composition = True

    def __init__(self,
                 llm_config=None,
                 ce_weight=1.0,
                 bce_weight=2.0,
                 dice_weight=0.5,
                 l1_weight=1.0,
                 iou_weight=1.0,
                 out_dim=256,
                 seg_token_idx=32007,
                 loc_token_idx=32008,
                 train_mask_decoder=True,
                 **kwargs
                 ):
        super(UllavaConfig, self).__init__(**kwargs)

        self.llm_config = UllavaCoreConfig(**llm_config) if llm_config else {}
        self.ce_weight = ce_weight
        self.bce_weight = bce_weight
        self.out_dim = out_dim
        self.dice_weight = dice_weight
        self.l1_weight = l1_weight
        self.iou_weight = iou_weight
        self.seg_token_idx = seg_token_idx
        self.loc_token_idx = loc_token_idx
        self.train_mask_decoder = train_mask_decoder

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["llm_config"] = self.llm_config.to_dict() if self.llm_config else {}
        output["ce_weight"] = self.ce_weight
        output["bce_weight"] = self.bce_weight
        output["dice_weight"] = self.dice_weight
        output["iou_weight"] = self.iou_weight
        output["l1_weight"] = self.l1_weight
        output["out_dim"] = self.out_dim
        output["seg_token_idx"] = self.seg_token_idx
        output["loc_token_idx"] = self.loc_token_idx
        output["train_mask_decoder"] = self.train_mask_decoder
        output["model_type"] = self.__class__.model_type
        return output


@registry.register_model('ullava')
class UllavaForCausalLM(PreTrainedModel):
    config_class = UllavaConfig

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        llm_config = config.llm_config
        self.llm = UllavaCoreForCausalLM(llm_config)
        # initialize sam and projector without checkpoint
        self.seg_projector, self.visual_model = self.init_seg_modules(llm_config.hidden_size)
        self.det_projector, self.det_decoder = self.init_det_modules(llm_config.hidden_size)

    def init_det_modules(self, hidden_size):
        # projector
        in_dim, out_dim = hidden_size, self.config.out_dim
        projector = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        )
        projector.train()
        for param in projector.parameters():
            param.requires_grad = True

        decoder = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim // 2, 4),
        )
        decoder.train()
        for param in decoder.parameters():
            param.requires_grad = True

        return projector, decoder

    def init_seg_modules(self, hidden_size):
        # Projection layer
        in_dim = hidden_size
        out_dim = self.config.out_dim
        projector = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        )
        projector.train()
        for param in projector.parameters():
            param.requires_grad = True

        # SAM
        visual_model = build_sam_vit_h(checkpoint=None)
        for param in visual_model.parameters():
            param.requires_grad = False
        if self.config.train_mask_decoder:
            visual_model.mask_decoder.train()
            for param in visual_model.mask_decoder.parameters():
                param.requires_grad = True

        return projector, visual_model

    def load_visual_checkpoint(self, checkpoint):
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        self.visual_model.load_state_dict(state_dict, strict=False)

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(
            self,
            images_sam: torch.FloatTensor,
            images: torch.FloatTensor,
            input_ids: torch.LongTensor,
            labels: torch.LongTensor,
            attention_mask: torch.LongTensor,
            mask_list: List[torch.FloatTensor],
            size_list: List[torch.Tensor],
            resize_list: List[tuple],
            bbox_list: List[torch.FloatTensor],
            inference: bool = False
    ):
        batch_size = input_ids.shape[0]

        image_embeddings = self.get_visual_embs(images_sam)
        seg_token_mask = input_ids[:, 1:] == self.config.seg_token_idx
        loc_token_mask = input_ids[:, 1:] == self.config.loc_token_idx

        # FIXME: [torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(), seg_token_mask]
        seg_token_mask = torch.cat(
            [seg_token_mask, torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda()], dim=1,
        )
        loc_token_mask = torch.cat(
            [loc_token_mask, torch.zeros((loc_token_mask.shape[0], 1)).bool().cuda()], dim=1,
        )

        output = self.llm.forward(
            images=images,
            attention_mask=attention_mask,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
        )

        output_hidden_states = output.hidden_states

        # seg, one image paired with several sentences
        last_hidden_state = self.seg_projector(output_hidden_states[-1])  # [bs, token_len, out_dim]
        pred_embeddings = last_hidden_state[seg_token_mask]  # [bs * num_sentence, out_dim]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs]
        seg_token_offset = seg_token_counts.cumsum(-1)  # [bs] e.g., [3, 6, 9, 12, 15, 18, 21]

        # loc
        last_hidden_state = self.det_projector(output_hidden_states[-1])  # [bs, token_len, out_dim]
        pred_loc_embeddings = last_hidden_state[loc_token_mask]  # [bs * num_sentence, out_dim]
        loc_token_counts = loc_token_mask.int().sum(-1)  # [bs]
        loc_token_offset = loc_token_counts.cumsum(-1)  # [bs] e.g., [3, 6, 9, 12, 15, 18, 21]

        # [bs + 1] e.g., [0, 3, 6, 9, 12, 15, 18, 21]
        seg_token_offset = torch.cat(
            [
                torch.zeros(1).long().cuda(),
                seg_token_offset
            ], dim=0
        )
        loc_token_offset = torch.cat(
            [
                torch.zeros(1).long().cuda(),
                loc_token_offset
            ], dim=0
        )

        pred_embeddings_, pred_loc_embeddings_ = [], []
        raw_size_list, resize_size_list = [], []
        for i in range(batch_size):
            # get seg embedding
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
            # get loc embedding
            start_i, end_i = loc_token_offset[i], loc_token_offset[i + 1]
            pred_loc_embeddings_.append(pred_loc_embeddings[start_i:end_i])

            resize_size_list.append(resize_list[i])
            raw_size_list.append(size_list[i])
        pred_embeddings, pred_loc_embeddings = pred_embeddings_, pred_loc_embeddings_

        multimask_output = False
        pred_masks, pred_boxes = [], []
        for i in range(batch_size):
            sparse_embeddings, dense_embeddings = \
                self.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),  # [num_image, 1, out_dim]
                )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)

            low_res_masks, iou_predictions = self.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_size_list[i],
                original_size=raw_size_list[i],
            )
            pred_masks.append(pred_mask[:, 0])

            pred_box = self.det_decoder(pred_loc_embeddings[i])
            pred_boxes.append(pred_box)

        gt_masks = mask_list
        gt_boxes = bbox_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "pred_boxes": pred_boxes,
                "gt_masks": gt_masks,
                "gt_boxes": gt_boxes,
                'logits': output.logits,
            }

        # next word loss
        ce_loss = output.loss
        ce_loss = ce_loss * self.config.ce_weight
        loss = ce_loss

        # mask loss
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0

        # box loss
        box_l1_loss = 0
        box_giou_loss = 0
        num_boxes = 0

        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                    gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                    sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
            )
            mask_dice_loss += (
                    dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

            gt_box = gt_boxes[batch_idx]
            pred_box = pred_boxes[batch_idx]
            assert (
                    gt_box.shape[0] == pred_box.shape[0]
            ), "gt_box.shape: {}, pred_box.shape: {}".format(
                gt_box.shape, pred_box.shape
            )
            box_l1_loss += bbox_l1_loss(pred_box, gt_box, gt_box.shape[0])
            box_giou_loss += bbox_giou_loss(pred_box, gt_box, gt_box.shape[0])
            num_boxes += gt_box.shape[0]

        mask_bce_loss = self.config.bce_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.config.dice_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        box_l1_loss = self.config.l1_weight * box_l1_loss / (num_boxes + 1e-8)
        box_giou_loss = self.config.iou_weight * box_giou_loss / (num_boxes + 1e-8)
        bbox_loss = box_l1_loss + box_giou_loss

        loss += mask_loss
        loss += bbox_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "bbox_loss": bbox_loss
        }

    def evaluate(
            self,
            images_sam,
            images,
            input_ids,
            raw_size_list,
            resize_list,
            max_new_tokens=32,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            no_repeat_ngram_size=None,
            stopping_criteria=None,
    ):
        with torch.inference_mode():
            outputs = self.llm.generate(
                input_ids=input_ids,
                images=images,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                top_p=top_p,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
                no_repeat_ngram_size=no_repeat_ngram_size,
                stopping_criteria=stopping_criteria
            )

            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.config.seg_token_idx
            loc_token_mask = output_ids[:, 1:] == self.config.loc_token_idx

            last_hidden_state = self.seg_projector(output_hidden_states[-1])  # [1, token_len, out_dim]
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            last_hidden_state = self.det_projector(output_hidden_states[-1])  # [1, token_len, out_dim]
            pred_loc_embeddings = last_hidden_state[loc_token_mask]
            loc_token_counts = loc_token_mask.int().sum(-1)  # [bs, ]
            loc_token_offset = loc_token_counts.cumsum(-1)
            loc_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), loc_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])

            pred_loc_embeddings_ = []
            for i in range(len(loc_token_offset) - 1):
                start_i, end_i = loc_token_offset[i], loc_token_offset[i + 1]
                pred_loc_embeddings_.append(pred_loc_embeddings[start_i:end_i])

            pred_embeddings, pred_loc_embeddings = pred_embeddings_, pred_loc_embeddings_

            image_embeddings = self.get_visual_embs(images_sam)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=raw_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

            pred_boxes = []
            for pred_loc_embedding in pred_loc_embeddings:
                pred_box = self.det_decoder(pred_loc_embedding)
                pred_boxes.append(pred_box)

        return output_ids, pred_masks, pred_boxes


AutoConfig.register("ullava", UllavaConfig)
AutoModelForCausalLM.register(UllavaConfig, UllavaForCausalLM)
