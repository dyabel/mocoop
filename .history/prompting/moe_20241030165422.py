import os.path as osp
from glob import glob
from tqdm import tqdm
import json
import itertools
import wandb
import torch.distributed as dist

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.nn.functional as F
# from prompting.lasp import PromptLearner
from collections import OrderedDict

from .losses import moe_contrastive_loss, contrastive_loss
import random
import collections

_tokenizer = _Tokenizer()




def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # print(prompts.shape, tokenized_prompts.shape)
        """
        # 假设 prompts 和 tokenized_prompts 是输入张量
        batch_size = 100
        num_batches = (prompts.size(0) + batch_size - 1) // batch_size

        # 初始化一个空列表来存储所有批次的结果
        all_results = []

        for i in range(num_batches):
            # print(i)
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, prompts.size(0))
            
            # 提取当前批次的数据
            prompts_batch = prompts[start_idx:end_idx]
            tokenized_prompts_batch = tokenized_prompts[start_idx:end_idx]
            
            # 处理当前批次的数据
            x = prompts_batch + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # 从 eot embedding 中提取 features（eot_token 是每个序列中的最大值）
            eot_indices = tokenized_prompts_batch.argmax(dim=-1)
            batch_result = x[torch.arange(x.shape[0]), eot_indices] @ self.text_projection

            # 将当前批次的结果添加到 all_results 中
            all_results.append(batch_result)

            # 将所有批次的结果合并成一个 tensor
        final_result = torch.cat(all_results, dim=0)
        return final_result
        """
        # """
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
        # """

def increase_top2_logits(logits, increment=1.0):
    """
    直接在 top-2 logits 上加上一个常数。
    
    参数:
        logits (torch.Tensor): 输入的 logits 向量。
        increment (float): 增加的值。
    
    返回:
        torch.Tensor: 修改后的 logits。
    """
    _, top2_indices = torch.topk(logits, k=2)  # 获取最高的两个logits的索引
    # print(top2_indices)
    increment_tensor = torch.zeros_like(logits)
    increment_tensor.scatter_(1, top2_indices, increment)

    logits += increment_tensor
    return logits


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, text_encoder_model, all_classnames):
        super().__init__()
        n_cls = len(all_classnames) if cfg.DATASET.INCLUDE_ALL_CLASSES else len(classnames)
        
        cumulative_sum = [0] + list(itertools.accumulate(cfg.TRAINER.MoCoOp.GROUPS))[:-1]
        ctx_init = cumulative_sum
        self.ctx_init = ctx_init
        self.class_token_position = []
        self.groups = cfg.TRAINER.MoCoOp.GROUPS
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        self.cfg = cfg
            
        self.n_ctx_before = []
        self.n_ctx_after = []
        prompt_template_before_all = [prompt.replace('.','').split('{}')[0][:-1] for prompt in cfg.TRAINER.MoCoOp.PROMPTS]
        prompt_template_after_all= [prompt.replace('.', '').split('{}')[1] for prompt in cfg.TRAINER.MoCoOp.PROMPTS]
        self.num_experts = len(prompt_template_before_all)
        if ctx_init:
            # use given words to initialize context vectors
            self.num_experts = len(ctx_init)
            for i, idx in enumerate(ctx_init):
                if len(prompt_template_before_all[idx]) == 0:
                    n_ctx = 0
                else:
                    n_ctx = len(prompt_template_before_all[idx].split(" "))
                prompt_template_before = prompt_template_before_all[idx].replace("_", " ")
                prompt = clip.tokenize(prompt_template_before)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors_before = embedding[0, 1 : 1 + n_ctx, :]
                self.register_parameter(f'ctx_before_{i}', nn.Parameter(ctx_vectors_before))
                self.n_ctx_before.append(n_ctx)
                if len(prompt_template_after_all[idx]) == 0:
                    n_ctx = 0
                else:
                    n_ctx = len(prompt_template_after_all[idx].split(" "))
                prompt_template_after = prompt_template_after_all[idx].replace("_", " ")
                prompt = clip.tokenize(prompt_template_after)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors_after = embedding[0, 1 : 1 + n_ctx, :]
                self.register_parameter(f'ctx_after_{i}', nn.Parameter(ctx_vectors_after))
                self.n_ctx_after.append(n_ctx)
                
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        classnames = [name.replace("_", " ") for name in classnames]
        all_classnames = [name.replace("_", " ") for name in all_classnames]

        if cfg.DATASET.INCLUDE_ALL_CLASSES:
            # Preserve class order
            classes_delta = [name for name in all_classnames if name not in classnames]
            print(f'Number of extra class names: {len(classes_delta)}')
            classnames += classes_delta
            print(f'Number of class names after: {len(classnames)}')
        
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        all_name_lens = [len(_tokenizer.encode(name)) for name in all_classnames]
        
        self.tokenized_prompts_all_experts = []
        for i in range(self.num_experts):
            prompts = [prompt_template_before_all[ctx_init[i]] + ' ' + name + prompt_template_after_all[ctx_init[i]] for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            self.tokenized_prompts_all_experts.append(tokenized_prompts)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
                # These token vectors will be saved when in save_model(),
                # but they should be ignored in load_model() as we want to use
                # those computed using the current class names
            self.register_buffer(f"token_prefix_{i}", embedding[:, :1, :])  # SOS
            self.register_buffer(f"token_suffix_{i}", embedding[:, 1 + self.n_ctx_before[i] :, :])  # CLS, EOS

        if cfg.TRAINER.MoCoOp.ENABLE:
            if self.cfg.DATASET.SUBSAMPLE_CLASSES == 'base':
                self.construct_references_lasp(cfg, clip_model, text_encoder_model, all_classnames, prompt_template_before_all, prompt_template_after_all, dtype, n_ctx)
            else:
                self.construct_references_lasp(cfg, clip_model, text_encoder_model, classnames, prompt_template_before_all, prompt_template_after_all, dtype, n_ctx)

        self.gate = nn.Linear(vis_dim, self.num_experts)
        # self.gate = nn.Sequential(OrderedDict([
        #     ("linear1", nn.Linear(vis_dim, vis_dim // 2)),
        #     ("relu", nn.ReLU(inplace=True)),
        #     ("linear2", nn.Linear(vis_dim // 2, self.num_experts))
        # ]))
        # self.noise_linear = nn.Linear(vis_dim, self.num_experts)


        self.n_cls = n_cls
        self.name_lens = name_lens
        self.all_name_lens = all_name_lens
        self.all_classnames = all_classnames
        self.classnames = classnames

        if cfg.TRAINER.MoCoOp.ENABLE_CORRECTION:
            # self.w = nn.Parameter(torch.zeros(self.num_experts, ctx_dim, device=embedding.device, dtype=dtype), requires_grad=self.cfg.TRAINER.MoCoOp.TRAIN_W)
            self.w = nn.Parameter(torch.zeros(1, ctx_dim, device=embedding.device, dtype=dtype), requires_grad=self.cfg.TRAINER.MoCoOp.TRAIN_W)


    def construct_references_lasp(self, cfg, clip_model, text_encoder_model, all_classnames, prompt_prefixs, prompt_suffixs, dtype, n_ctx):
        # template_prompts = cfg.TRAINER.MoCoOp.CTX_INIT
        all_classnames = [name.replace("_", " ") for name in all_classnames]

        all_class_text_features = []
        for i in range(len(prompt_prefixs)):
            prompts = [prompt_prefixs[i] + ' ' +  name + prompt_suffixs[i] + '.' for name in all_classnames]
            # prompts = [c_init + " " + name + "." for name in all_classnames]
            tokenized_prompts_all_c = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            text_encoder_model.cuda()
            with torch.no_grad():
                embedding_all_cls = clip_model.token_embedding(tokenized_prompts_all_c).cuda().type(dtype)
                class_text_features = text_encoder_model(embedding_all_cls, tokenized_prompts_all_c).type(dtype)
                all_class_text_features.append(class_text_features)
            
        class_text_features = torch.stack(all_class_text_features, dim=0)
        class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)
        self.register_buffer("class_text_features", class_text_features)
        grouped_prototypes = torch.stack([group.mean(dim=0) for group in torch.split(class_text_features, self.groups)])
        grouped_prototypes = grouped_prototypes / grouped_prototypes.norm(dim=-1, keepdim=True)
        self.register_buffer("grouped_prototypes", grouped_prototypes)


        self.tokenized_prompts_all_class = []
        for i in range(self.num_experts):
            prompts = [prompt_prefixs[self.ctx_init[i]] + " " + name + prompt_suffixs[self.ctx_init[i]] + '.' for name in all_classnames]
            tokenized_prompts_all_c_ = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            self.tokenized_prompts_all_class.append(tokenized_prompts_all_c_)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts_all_c_).type(dtype)

            self.register_buffer(f"token_prefix_all_{i}", embedding[:, :1, :])  # SOS
            self.register_buffer(f"token_suffix_all_{i}", embedding[:, 1 + self.n_ctx_before[i]:, :])  # CLS, EOS

        # self.ref_tokenized_prompts_all = tokenized_prompts_all_c
        self.n_cls_all = len(prompts)
    
    def construct_prompts(self, n_ctx_after, ctx_before, ctx_after, prefix, suffix, name_lens, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        """
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )
        """
        
       
        prompts = []
        for i in range(len(prefix)):
            name_len = name_lens[i]
            prefix_i = prefix[i : i + 1, :, :]
            class_i = suffix[i : i + 1, :name_len, :]
            # print(i, len(prefix), len(suffix), len(suffix[i]), name_len+self.n_ctx_after[i])
            suffix_i = suffix[i : i + 1, name_len+n_ctx_after:, :]
            # print(ctx_before.shape)
            # print(prefix_i.shape, ctx_before.shape, class_i.shape, ctx_after.shape, suffix_i.shape)
            prompt = torch.cat(
                [
                    prefix_i,     # (1, 1, dim)
                    ctx_before[i].unsqueeze(0),  # (1, n_ctx//2, dim)
                    class_i,      # (1, name_len, dim)
                    ctx_after[i].unsqueeze(0),  # (1, n_ctx//2, dim)
                    suffix_i,     # (1, *, dim)
                ],
                dim=1,
            )
            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)

       
        return prompts

 
    
    
    def forward(self, all=False):
        # prompts_all = []
        #         if not all:
        prompts = []
        for i in range(self.num_experts):
            ctx_before = getattr(self, f'ctx_before_{i}')
            ctx_after = getattr(self, f'ctx_after_{i}')
            # ctx_all_experts.append(ctx)
            # Use instance-conditioned context tokens for all classes
            if not all:
                prefix = getattr(self, f"token_prefix_{i}")
                suffix = getattr(self, f"token_suffix_{i}")
                n_cls = self.n_cls
                name_lens = self.name_lens
            else:
                prefix = getattr(self, f"token_prefix_all_{i}")
                suffix = getattr(self, f"token_suffix_all_{i}")
                n_cls = len(self.all_classnames)
                name_lens = self.all_name_lens  
            ctx_before_i = ctx_before.unsqueeze(0).expand(n_cls, -1, -1)
            ctx_after_i = ctx_after.unsqueeze(0).expand(n_cls, -1, -1)
            pts_i = self.construct_prompts(self.n_ctx_after[i], ctx_before_i, ctx_after_i, prefix, suffix, name_lens)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)

            
        # for batch_idx in range(im_features.shape[0]):           
        #     prompts = []
        #     for i in indices[batch_idx]:
        #         if not all:
        #             prefix = getattr(self, f"token_prefix_{i}")
        #             suffix = getattr(self, f"token_suffix_{i}")
        #             n_cls = self.n_cls
        #         else:
        #             prefix = getattr(self, f"token_prefix_all_{i}")
        #             suffix = getattr(self, f"token_suffix_all_{i}")
        #             n_cls = len(self.all_classnames)
        #         ctx_i = ctx_all_experts[i].unsqueeze(0).expand(n_cls, -1, -1)
        #         pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
        #         prompts.append(pts_i)
        #     prompts = torch.stack(prompts)
        #     prompts_all.append(prompts)

        return prompts



class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, all_classnames, device):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.device = device
        # self.text_encoder = TextEncoder(clip_model)
        self.text_encoder = nn.DataParallel(TextEncoder(clip_model))
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model, self.text_encoder, all_classnames).to(clip_model.dtype)
        self.tokenized_prompts_all_experts = self.prompt_learner.tokenized_prompts_all_experts
        self.grouped_prototypes = self.prompt_learner.grouped_prototypes
        self.group_features = self.grouped_prototypes.mean(dim=1)
        self.group_features = self.group_features / self.group_features.norm(dim=-1, keepdim=True)
        if cfg.TRAINER.MoCoOp.ENABLE:
            self.tokenized_prompts_all_class = self.prompt_learner.tokenized_prompts_all_class
            self.num_templates = len(self.prompt_learner.class_text_features)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cfg = cfg
        self.n_cls_all = len(all_classnames)
        self.n_cls = len(classnames)
        self.num_experts = self.prompt_learner.num_experts
        # self.loss = contrastive_loss
        self.loss = moe_contrastive_loss
        self.gate_count = collections.defaultdict(int)
        self.eps = 0.1
        self.max_iter = 100
        self.groups = cfg.TRAINER.MoCoOp.GROUPS
        self.dataset = cfg.DATASET.NAME


    def compute_gated_text_feature(self, image_features, tokenized_prompts_all, all=False, text_gating=None):
        expert_text_features_all = []
        if text_gating is None:
            gating_distribution = self.prompt_learner.gate(image_features)
        else:
            gating_distribution = text_gating
        # logits = gating_distribution
        # logits = self.prompt_learner.gate(image_features)
        # noise_logits = self.prompt_learner.noise_linear(image_features)
        # noise = torch.randn_like(logits)*F.softplus(noise_logits)
        # gating_distribution = logits + noise
        topk_gating_distribution, indices = torch.topk(gating_distribution, k=2)
        # for i in range(len(indices)):
        #     for j in indices[i]:
        #         self.gate_count[j.item()] += 1
        # print(self.gate_count)
        prompts = self.prompt_learner()
        
        for i, prompt in enumerate(prompts):
            if self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE < 4 and i not in indices:
                text_features_per_expert = torch.zeros(self.n_cls, self.text_encoder.module.ln_final.weight.shape[0], device=image_features.device, dtype=image_features.dtype)
            else:
                tokenized_prompts = tokenized_prompts_all[i]
                text_features_per_expert = self.text_encoder(prompt, tokenized_prompts)
                if self.cfg.TRAINER.MoCoOp.ENABLE_CORRECTION:
                    w = self.prompt_learner.w
                    text_features_per_expert = text_features_per_expert + w
                text_features_per_expert = text_features_per_expert / text_features_per_expert.norm(dim=-1, keepdim=True)
            expert_text_features_all.append(text_features_per_expert)
        text_features_all = []
        for batch_idx in range(len(image_features)):
            text_features = []
            for i in range(len(indices[batch_idx])):
                text_features.append(expert_text_features_all[indices[batch_idx][i]])
            text_features = torch.stack(text_features)
            text_features_all.append(text_features)
            
        topk_gating_distribution = torch.softmax(topk_gating_distribution, dim=1)
        # print(topk_gating_distribution.shape, torch.stack(text_features_all).shape)
        text_features = torch.einsum("bk,bkcd->bcd", topk_gating_distribution, torch.stack(text_features_all))
        return text_features, gating_distribution, indices, torch.stack(expert_text_features_all)
        # return text_features_per_expert, gating_distribution

    def forward_text_to_text(self, text_features=None, indices=None):
        with torch.no_grad():
            class_text_features = self.prompt_learner.class_text_features
            # class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)

        if torch.rand(1).item() < 0.5:
            noise = 0.05 * torch.randn_like(class_text_features)
            class_text_features.add_(noise)
        # """
        if self.cfg.DATASET.SUBSAMPLE_CLASSES == 'base':
            prompts = self.prompt_learner(all=True)
            expert_text_features_all = []
            for i, prompt in enumerate(prompts):
                if self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE < 4 and i not in indices:
                    text_features_per_expert = torch.zeros(self.n_cls_all, self.text_encoder.module.ln_final.weight.shape[0], device=self.device, dtype=self.dtype)
                else:
                    tokenized_prompts = self.tokenized_prompts_all_class[i]
                    # print(self.text_encoder.module.ln_final.weight.device, prompt.device)
                    text_features_per_expert = self.text_encoder(prompt, tokenized_prompts)

                    if self.cfg.TRAINER.MoCoOp.ENABLE_CORRECTION:
                        w = self.prompt_learner.w
                        # w = self.prompt_learner.w[i].unsqueeze(0)
                        # print(text_features_per_expert.shape, w.shape)
                        if len(text_features_per_expert.shape) == 2:
                            text_features_per_expert = text_features_per_expert + w
                        elif len(text_features_per_expert.shape) == 3:
                            text_features_per_expert = text_features_per_expert + w.unsqueeze(0)
                    

                    text_features_per_expert = text_features_per_expert / text_features_per_expert.norm(dim=-1, keepdim=True)
                expert_text_features_all.append(text_features_per_expert)
            text_features = torch.stack(expert_text_features_all)
        # else:
            # assert self.prompt_learner.all_classnames == self.prompt_learner.classnames
        # """
        # text_features = text_features.unsqueeze(0)

        label = torch.arange(self.prompt_learner.n_cls_all, device=class_text_features.device, dtype=torch.long).unsqueeze(0).expand(len(text_features), -1)

        # return sum([self.loss(text_features, class_text_features, label, t=self.logit_scale)[0] for text_features in expert_text_features_all])/len(expert_text_features_all)
        # label = torch.arange(self.prompt_learner.n_cls_all, device=class_text_features.device, dtype=torch.long).unsqueeze(0).expand(2, -1)
     
        # print(text_features.shape, self.grouped_prototypes.shape)
        assert text_features.shape == self.grouped_prototypes.shape
        loss, _, group_features = self.loss(text_features, self.grouped_prototypes, label, t=self.logit_scale, groups=self.groups, indices=indices)
        return loss, group_features
    
    

    def forward(self, image, label=None):

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        tokenized_prompts_all_experts = self.tokenized_prompts_all_experts

        text_features, gating_distribution, indices, expert_text_features = self.compute_gated_text_feature(image_features, tokenized_prompts_all_experts)
        # """
        if self.cfg.TRAINER.MoCoOp.ENABLE:
            p = 1.
            if random.random() < p:
                loss_text, group_features = self.forward_text_to_text(expert_text_features, indices=indices)
            else:
                loss_text = 0
            # group_features = F.normalize(group_features, dim=-1)
            if label is not None:
                group_features = self.group_features[:, label]
                text_gating = torch.einsum('bd, gbd->bg', image_features, group_features)
                text_gating = increase_top2_logits(text_gating, increment=10.0)
        # """
        loss, logits = self.loss(image_features, text_features, label, t=self.logit_scale, instance_loss=True)

        if self.prompt_learner.training:
            if self.cfg.TRAINER.MoCoOp.ENABLE:
                loss += self.cfg.TRAINER.MoCoOp.TEXT_LOSS_WEIGHT * loss_text / p

        if self.prompt_learner.training:
            if self.cfg.TRAINER.MoCoOp.ENABLE:
                return loss  + self.cfg.TRAINER.MoCoOp.GATE_LOSS_WEIGHT * F.cross_entropy(gating_distribution, F.softmax(text_gating, dim=1))
            else:
                return loss
        else:
            return logits


@TRAINER_REGISTRY.register()
class MoCoOp(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MoCoOp.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        if cfg.DATASET.NAME in ['OxfordPets']:
            all_classnames = self.dm.dataset.all_classnames
        elif cfg.DATASET.NAME in ['ImageNetR', 'ImageNetA', 'ImageNetV2', 'ImageNetSketch']:
            all_classnames = self.dm.dataset.classnames
        else:
            all_classnames = self.dm.dataset.all_class_names

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.MoCoOp.PREC == "fp32" or cfg.TRAINER.MoCoOp.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, all_classnames, self.device)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)

        if cfg.TRAINER.MoCoOp.FINETUNE_VIT_LN:
            print('Re-enabling LN...')
            for name, param in self.model.named_parameters():
                if 'image_encoder' in name and ('ln_2' in name or 'ln_1' in name):
                    param.requires_grad_(True)  
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        # rank = dist.get_rank()
        self.model.to(self.device)
        # print(rank)
        # self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[rank], find_unused_parameters=True)
        # NOTE: only give prompt_learner to the optimizer
        # self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        # self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        # self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        if cfg.TRAINER.MoCoOp.FINETUNE_VIT_LN:
            group1, group2 = [], []
            for name, param in self.model.named_parameters():
                if 'image_encoder' in name and ('ln_2' in name or 'ln_1' in name):
                    group1.append(param)
                else:
                    group2.append(param)

            param_groups = [
                {
                    "params": group1,
                    "lr": cfg.OPTIM.LR * 0.1
                },
                {
                    "params": group2
                },
            ]
            self.optim = build_optimizer(self.model, cfg.OPTIM, param_groups=param_groups)
        else:
            self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MoCoOp.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.MoCoOp.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        if isinstance(input, list):
            input = [inp.to(self.device, non_blocking=True) for inp in input]
        else:
            input = input.to(self.device, non_blocking=True)
        label = label.to(self.device)

        if self.cfg.DATALOADER.K_TRANSFORMS > 1:
            input = torch.cat(input)
            label = label.repeat(self.cfg.DATALOADER.K_TRANSFORMS)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                print('Model not found at "{}", retrying to find one automatically...'.format(model_path))
                model_path = glob(f'{directory}/{name}/model-best.pth.tar-*')[0]
                if not osp.exists(model_path):
                    raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            ignore_list = ['token_prefix', 'token_suffix', 'token_prefix_all', 'token_suffix_all', 'class_text_features', 'grouped_prototypes']
            ignore_list += [f'prompt_learner.{il}' for il in ignore_list]

            for k in ignore_list:
                state_dict.pop(k, None)
            for  key in list(state_dict.keys()):
                if "token_prefix" in key:
                    del state_dict[key]
                if "token_suffix" in key:
                    del state_dict[key]
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            w_weights = None
            new_state_dict = {}
            for k, v in state_dict.items():
                if k in self._models[name].state_dict():
                    # if k == 'w':
                    #     w_weights = v
                    if v.size() == self._models[name].state_dict()[k].size():
                        new_state_dict[k] = v
                    else:
                        print(k, v.shape, self._models[name].state_dict()[k].size())
            print(f'Num of preserved keys: {len(new_state_dict)}')
            print(f'Keys: {new_state_dict.keys()}')
            #new_state_dict = {}
            self._models[name].load_state_dict(new_state_dict, strict=False)
        return w_weights
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
            # if dist.get_rank() == 0:
            wandb.run.summary[tag] = v
            print(tag, v, file=open('result.txt', 'a'))

        with open(osp.join(self.output_dir, 'results.json'), 'w') as fp:
            json.dump(results, fp)

        return list(results.values())[0]
        
