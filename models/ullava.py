"""
u-LLaVA with segmentation module
SAM patch is Adapted form: https://github.com/dvlab-research/LISA/blob/main/model/LISA.py
"""

import copy
import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
from utils.registry import registry
from models.segment_anything import build_sam_vit_h
from models.ullava_core import UllavaCoreConfig, UllavaCoreForCausalLM
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig, AutoConfig


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        scale=1000,
        eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks:
        scale:
        eps:
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks:
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class UllavaConfig(PretrainedConfig):
    model_type = "ullava"
    is_composition = True

    def __init__(self,
                 llm_config=None,
                 ce_weight=0.5,
                 bce_weight=0.5,
                 dice_weight=-1,
                 out_dim=256,
                 seg_token_idx=32007,
                 train_mask_decoder=True,
                 **kwargs
                 ):
        super(UllavaConfig, self).__init__(**kwargs)

        self.llm_config = UllavaCoreConfig(**llm_config) if llm_config else {}
        self.ce_weight = ce_weight
        self.bce_weight = bce_weight
        self.out_dim = out_dim
        self.dice_weight = dice_weight
        self.seg_token_idx = seg_token_idx
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
        output["out_dim"] = self.out_dim
        output["seg_token_idx"] = self.seg_token_idx
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
        self.visual_model, self.text_hidden_fcs = self.init_seg_modules(llm_config.hidden_size)

    def init_seg_modules(self, hidden_size):
        # SAM
        visual_model = build_sam_vit_h(checkpoint=None)
        for param in visual_model.parameters():
            param.requires_grad = False
        if self.config.train_mask_decoder:
            visual_model.mask_decoder.train()
            for param in visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = hidden_size
        out_dim = self.config.out_dim
        text_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        )
        text_fcs.train()
        for param in text_fcs.parameters():
            param.requires_grad = True

        return visual_model, text_fcs

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
            inference: bool = False,
            **kwargs
    ):
        batch_size = input_ids.shape[0]

        image_embeddings = self.get_visual_embs(images_sam)
        seg_token_mask = input_ids[:, 1:] == self.config.seg_token_idx
        # FIXME: [torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(), seg_token_mask]
        seg_token_mask = torch.cat(
            [seg_token_mask, torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda()], dim=1,
        )

        output = self.llm.forward(
            images=images,
            attention_mask=attention_mask,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
        )

        output_hidden_states = output.hidden_states

        last_hidden_state = self.text_hidden_fcs(output_hidden_states[-1])  # [bs, token_len, out_dim]

        # one image paired with several sentences
        pred_embeddings = last_hidden_state[seg_token_mask]  # [bs * num_sentence, out_dim]

        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs]
        seg_token_offset = seg_token_counts.cumsum(-1)  # [bs] e.g., [3, 6, 9, 12, 15, 18, 21]

        # [bs + 1] e.g., [0, 3, 6, 9, 12, 15, 18, 21]
        seg_token_offset = torch.cat(
            [
                torch.zeros(1).long().cuda(),
                seg_token_offset
            ], dim=0
        )

        pred_embeddings_ = []
        raw_size_list = []
        resize_size_list = []
        for i in range(batch_size):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
            resize_size_list.append(resize_list[i])
            raw_size_list.append(size_list[i])
        pred_embeddings = pred_embeddings_

        multimask_output = False
        pred_masks = []
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

        gt_masks = mask_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                'logits': output.logits
            }

        ce_loss = output.loss
        ce_loss = ce_loss * self.config.ce_weight
        loss = ce_loss
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
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

        mask_bce_loss = self.config.bce_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.config.dice_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss += mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss
        }

    def evaluate(
            self,
            images_sam,
            images,
            input_ids,
            raw_size_list,
            resize_list,
            max_new_tokens=32,
            stopping_criteria=None,
    ):
        with torch.inference_mode():
            outputs = self.llm.generate(
                input_ids=input_ids,
                images=images,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                # do_sample=True,
                # temperature=0.2,
                output_hidden_states=True,
                return_dict_in_generate=True,
                stopping_criteria=stopping_criteria
            )

            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.config.seg_token_idx
            # print(output_hidden_states.size(), seg_token_mask.size())

            last_hidden_state = self.text_hidden_fcs(output_hidden_states[-1])  # [1, token_len, out_dim]

            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

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

        return output_ids, pred_masks


AutoConfig.register("ullava", UllavaConfig)
AutoModelForCausalLM.register(UllavaConfig, UllavaForCausalLM)
