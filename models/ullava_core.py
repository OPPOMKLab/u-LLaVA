"""
Core of uLLaVA

    -------------------------
       UllavaCoreForCausalLM    ----- Core (Extend LLaVA, supports Image/Video Caption, VQA.)
    -------------------------
                 |
    -------------------------
        UllavaForCausalLM       ----- Core with Segmentation module
    -------------------------

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

import copy
import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional, List
from utils.registry import registry
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaModel, LlamaForCausalLM, CLIPVisionModel, \
    CLIPVisionConfig, LlamaConfig, AutoConfig, AutoModelForCausalLM


class UllavaCoreConfig(LlamaConfig):
    model_type = "ullava_core"
    is_composition = True

    def __init__(self,
                 vision_config=None,
                 vision_hidden_layer=-1,
                 projector_from_scratch=True,
                 mm_token_ids=None,
                 **kwargs
                 ):
        super(UllavaCoreConfig, self).__init__(**kwargs)

        self.vision_hidden_layer = vision_hidden_layer
        self.mm_token_ids = mm_token_ids
        self.projector_from_scratch = projector_from_scratch

        self.vision_config = CLIPVisionConfig(**vision_config) if vision_config else {}

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict() if self.vision_config else {}
        output["vision_hidden_layer"] = self.vision_hidden_layer
        output["mm_token_ids"] = self.mm_token_ids
        output["projector_from_scratch"] = self.projector_from_scratch
        output["model_type"] = self.__class__.model_type
        return output


@registry.register_model('ullava_core')
class UllavaCoreForCausalLM(LlamaForCausalLM):
    config_class = UllavaCoreConfig

    def __init__(self, config: UllavaCoreConfig):
        """
        keep the same structure with LlamaForCausalLM: model + lm_head
        :param config:
            Evallama2Config:
                -llm_config
                -vision_config
                -vision_hidden_layer

        EvaLLaMA^2:
            causal_llm:
                model (LlamaModel)
                lm_head (MLP)
            vision_encoder: CLIP
            vision_projector: MLP
        """
        super(LlamaForCausalLM, self).__init__(config)

        self.config = config

        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.vision_encoder = CLIPVisionModel(config.vision_config)
        # projector from vision to LLM space: [1024, 4096]
        self.vision_projector = nn.Linear(config.vision_config.hidden_size, config.hidden_size)

        self.vision_hidden_layer = config.vision_hidden_layer
        self.projector_from_scratch = config.projector_from_scratch
        self.mm_token_ids = config.mm_token_ids

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def init_mm_tokens(self, tokenizer, mm_tokens):
        """
        fetch mm_token_id from tokenizer
        :return:
        """
        mm_token_ids = {k: tokenizer.convert_tokens_to_ids(v) for k, v in mm_tokens.items()}
        self.config.mm_token_ids = mm_token_ids
        self.mm_token_ids = self.config.mm_token_ids

    def encode_image(self, image_tensors):
        """
        :param image_tensors:
        :return:
        """
        with torch.no_grad():
            # For image: [bs, 3, 224, 224]
            image_forward_outs = self.vision_encoder(image_tensors, output_hidden_states=True)
            select_hidden_state = image_forward_outs.hidden_states[self.vision_hidden_layer]
            # remove CLS embedding [:, 1:]
            image_features = select_hidden_state[:, 1:]  # [bs, num_patches=16*16, 1024]

        return image_features

    def encode_video(self, video_clip_tensors):
        """
        :param video_clip_tensors: [bs, C, T, H, W]
        :return: temporal-spatial features
        """
        bs = video_clip_tensors.size(0)
        # For video: [bs, n_frm, 3, 224, 224] -> [bs * n_frm, 3, 224, 224]
        video_clip_tensors = rearrange(video_clip_tensors, 'b c t h w -> (b t) c h w')
        with torch.no_grad():
            video_forward_outs = self.vision_encoder(video_clip_tensors, output_hidden_states=True)
            select_hidden_state = video_forward_outs.hidden_states[self.vision_hidden_layer]
            # Remove CLS embedding [:, 1:]
            video_features = select_hidden_state[:, 1:]  # [bs * n_frm, num_patches=16*16, 1024]
        video_features = rearrange(video_features, '(b t) n d -> b t n d', b=bs)  # [bs, n_frm, num_patches, 1024]

        spatial_features = video_features.mean(dim=1)  # [bs, num_patches, 1024]
        temporal_features = video_features.mean(dim=2)  # [bs, n_frm, 1024]

        st_features = torch.concat([temporal_features, spatial_features], dim=1)  # [bs, (n_frm+num_patches), 1024]

        return st_features

    def embed_images_videos(self,
                            input_ids: torch.LongTensor = None,
                            images: Optional[torch.FloatTensor] = None,
                            videos: Optional[torch.FloatTensor] = None
                            ):

        if input_ids.shape[1] == 1:
            return input_ids, None

        inputs_embeds = self.model.embed_tokens(input_ids)

        # [bs, C, H, W]  -> [bs, num_patches, 1024]
        image_features = self.encode_image(images) if images is not None else None
        # [bs, C, T, H, W] -> [bs, n_frm + num_patches, 1024]
        video_features = self.encode_video(videos) if videos is not None else None

        batch_input_embeds = []
        cur_image_idx, cur_video_idx = 0, 0

        img_start_id, img_end_id, vid_start_id, vid_end_id = \
            self.mm_token_ids["IMG_START"], self.mm_token_ids["IMG_END"], \
            self.mm_token_ids["VID_START"], self.mm_token_ids["VID_END"]

        for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
            num_img_start, num_img_end = (cur_input_ids == img_start_id).sum(), (cur_input_ids == img_end_id).sum()
            num_vid_start, num_vid_end = (cur_input_ids == vid_start_id).sum(), (cur_input_ids == vid_end_id).sum()

            assert num_img_start == num_img_end and num_vid_start == num_vid_end, \
                print("Number of image start and end tokens should be the same. {0} vs {1}, {2} vs {3}".
                      format(num_img_start, num_img_end, num_vid_start, num_vid_end))

            if num_img_start == 0 and num_vid_start == 0:
                # if there are only texts in batch, for example SQA and Alpaca
                # if do not set this, will cause NCCL timeout (CUDA error)
                dummy_image_features = torch.zeros(256, 1024, device=inputs_embeds.device,
                                                   dtype=inputs_embeds.dtype)
                dummy_image_features = self.vision_projector(dummy_image_features)  # [bs, num_patch=16*16, 4096]
                cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                cur_new_input_embeds = cur_input_embeds

            elif num_img_start > 0:
                # if there are images in batch
                im_start_tokens = torch.where(cur_input_ids == img_start_id)[0]
                # im_start_tokens is a list and only one image in it
                img_start_pos = im_start_tokens[0]
                # [num_patch, 1024] -> [num_patch, 4096] .to(device=cur_input_embeds.device)
                cur_image_features = self.vision_projector(image_features[cur_image_idx])
                num_patch = cur_image_features.shape[0]

                if self.projector_from_scratch:
                    # FIXME: do this at pre-training according to LLaVA, failure to do so may result in performance drop
                    # FIXME: if you have a better solution, please concat us, thank you
                    cur_new_input_embeds = torch.cat((cur_input_embeds[:img_start_pos].detach(),
                                                      cur_input_embeds[img_start_pos:img_start_pos + 1],  # IM_START
                                                      cur_image_features,
                                                      cur_input_embeds[img_start_pos + num_patch + 1:
                                                                       img_start_pos + num_patch + 2],  # IM_END
                                                      cur_input_embeds[img_start_pos + num_patch + 2:].detach()),
                                                     dim=0)
                else:
                    # Fintuning stage, train LLM and visual projector, all embeddings shoule be trained
                    cur_new_input_embeds = torch.cat((cur_input_embeds[:img_start_pos + 1],
                                                      cur_image_features,
                                                      cur_input_embeds[img_start_pos + num_patch + 1:]), dim=0)
                cur_image_idx += 1

            elif num_vid_start > 0:
                # if there are videos in batch
                vid_start_tokens = torch.where(cur_input_ids == vid_start_id)[0]
                vid_start_pos = vid_start_tokens[0]
                # [n_frm+num_patch, 4096]
                cur_video_features = self.vision_projector(video_features[cur_video_idx])
                num_frm_patch = cur_video_features.shape[0]  # n_frm + num_patches

                if self.projector_from_scratch:
                    cur_new_input_embeds = torch.cat((cur_input_embeds[:vid_start_pos].detach(),
                                                      cur_input_embeds[vid_start_pos:vid_start_pos + 1],  # V_START
                                                      cur_video_features,
                                                      cur_input_embeds[vid_start_pos + num_frm_patch + 1:
                                                                       vid_start_pos + num_frm_patch + 2],  # V_END
                                                      cur_input_embeds[
                                                      vid_start_pos + num_frm_patch + 2:].detach()),
                                                     dim=0)
                else:
                    cur_new_input_embeds = torch.cat((cur_input_embeds[:vid_start_pos + 1],
                                                      cur_video_features,
                                                      cur_input_embeds[vid_start_pos + num_frm_patch + 1:]), dim=0)
                cur_video_idx += 1
            else:
                raise NotImplementedError
            batch_input_embeds.append(cur_new_input_embeds)

        # list -> tensor stack
        new_inputs_embeds = torch.stack(batch_input_embeds, dim=0)

        return None, new_inputs_embeds

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                images: Optional[torch.FloatTensor] = None,
                videos: Optional[torch.FloatTensor] = None,
                return_dict: Optional[bool] = None,
                ):
        """
        Args:
            input_ids: [bs, length, dim]
            attention_mask: [bs, length, dim]
            labels: [bs, length, dim]
            images: [bs, C, H, W]
            videos: [bs, C, T, H, W]
        :return: loss when training else None
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, inputs_embeds = self.embed_images_videos(input_ids, images, videos)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # if self.training:
        #     output_hidden_states = outputs.hidden_states
        # else:
        #     output_hidden_states = hidden_states

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor = None,
            inputs_embeds: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            images: torch.Tensor = None,
            videos: torch.Tensor = None,
            labels: torch.LongTensor = None,
            past_key_values=None,
            **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
                "videos": videos,
            }
        )
        return model_inputs


AutoConfig.register("ullava_core", UllavaCoreConfig)
AutoModelForCausalLM.register(UllavaCoreConfig, UllavaCoreForCausalLM)
