"""
Transformer with softmax-logit normalization.
Model family: GPT / lambda attention;
Feature selection layers configurable (with attention window mask).
Recurrent inference layers configurable.

For Lambda attention family, can customize hyperparams such as
    U/V split, Ignored U/V, and non-orthonormal penalty term in loss fn.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from dataclasses import dataclass
from lightning.fabric import is_wrapped
from torch.distributed.fsdp import FSDPModule
from typing import Any, Dict, List, Optional, Tuple, Union

from model_base import (
    ModelBase,
    FusedCrossEntropyLoss,
    LayerNorm,
    RotaryPositionEmbedding,
    SwiGLU,
    MaskedIndexINorm,
)


@dataclass
class NormalizedTransformerConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 384
    dtype: torch.dtype = torch.float16
    dropout: float = 0.0
    bias: bool = False
    context_length: int = 0
    use_fused: bool = False
    feature_selection_layers: int = 4
    feature_selection_heads: int = 4
    feature_selection_backward_distance: int = 7
    rope_enabled_for_inference: bool = True
    recurrent_inference: bool = False
    use_vanilla_attn: bool = True
    n_fixed_embeddings: int = 16
    n_model_split_u: int = 64
    n_model_split_v: int = 4
    orthogonal_regularization_factor: float = 0.0

class MLP(nn.Module):
    def __init__(self, config: NormalizedTransformerConfig):
        super().__init__()
        hidden_dim = 8 * config.n_embd / 3
        hidden_dim = 128 * round(hidden_dim / 128)
        self.up = SwiGLU(
            config.n_embd, hidden_dim, bias=config.bias, dtype=config.dtype
        )
        self.down = nn.Linear(
            hidden_dim, config.n_embd, bias=config.bias, dtype=config.dtype
        )
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.down(x)
        x = self.dropout(x)
        return x

class LambdaAttention(nn.Module):
    """Highly optimized Attention with decomposed U and V operation matrices and intent-feature casting."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        bias: bool,
        use_vanilla_attn: bool,
        n_fixed_embeddings: int,
        n_model_split_u: int,
        n_model_split_v: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by the attention head count")

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.sqrt_head_dim = math.sqrt(self.head_dim)
        self.sqrt_two_head_dim = math.sqrt(2 * self.head_dim)
        self.n_embd = n_embd
        self.n_fixed_embeddings = n_fixed_embeddings
        self.n_model_split_u = n_model_split_u
        self.n_model_split_v = n_model_split_v
        self.dtype = dtype
        self.use_vanilla_attn = use_vanilla_attn

        self.fixed_features = nn.Parameter(
            torch.empty(1, n_fixed_embeddings, n_embd, dtype=dtype)
        )

        # self.intent = nn.Linear(n_embd, n_embd, bias=bias, dtype=dtype)
        self.value_proj = None
        if self.use_vanilla_attn:
            self.query_proj = nn.Linear(n_embd, n_embd, bias=bias, dtype=dtype)
            self.key_proj = nn.Linear(n_embd, n_embd, bias=bias, dtype=dtype)
            self.value_proj = nn.Linear(n_embd, n_embd, bias=bias, dtype=dtype)
        else:
            self.intent_operator = nn.Linear(n_embd, n_embd, bias=bias, dtype=dtype)
            self.intent_operand = nn.Linear(n_embd, n_embd, bias=bias, dtype=dtype)
            self.characteristic_operator = nn.Linear(n_embd, n_embd, bias=bias, dtype=dtype)
            self.characteristic_operand = nn.Linear(n_embd, n_embd, bias=bias, dtype=dtype)
            self.feature_proj = nn.Linear(n_embd, n_embd, bias=bias, dtype=dtype)
            # Need to declare them as 2D to allow Muon to pick them up as params
            self.op_u = nn.Parameter(
                torch.empty(n_head * self.head_dim, n_model_split_u, dtype=dtype)
            )
            self.op_v = nn.Parameter(
                torch.empty(n_head * n_model_split_v, self.head_dim, dtype=dtype)
            )
        self.logit_norm = MaskedIndexINorm()
        self.norm_variance = nn.Parameter(
            torch.ones(n_head, dtype=dtype)
        )
        self.dropout = dropout
        self.resid_dropout = (
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias, dtype=dtype)

        nn.init.normal_(self.fixed_features, mean=0.0, std=1.0)
        if not self.use_vanilla_attn:
            nn.init.normal_(self.op_u, mean=0.0, std=1.0 / self.sqrt_head_dim)
            nn.init.normal_(self.op_v, mean=0.0, std=1.0 / self.sqrt_head_dim)

    def orthogonal_regularization_loss(self) -> torch.Tensor:
        loss = self.fixed_features.new_zeros(())
        if not self.use_vanilla_attn:
            u = self.op_u.reshape(
                self.n_head, self.head_dim, self.n_model_split_u
            ).float()
            identity_u = torch.eye(
                self.n_model_split_u, device=u.device, dtype=u.dtype
            )
            loss = loss + (u.transpose(-1, -2) @ u - identity_u).square().mean()
            
            v = self.op_v.reshape(
                self.n_head, self.n_model_split_v, self.head_dim
            ).float()
            identity_v = torch.eye(
                self.n_model_split_v, device=v.device, dtype=v.dtype
            )
            loss = loss + (v @ v.transpose(-1, -2) - identity_v).square().mean()
        return loss

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if self.use_vanilla_attn:
            return self.forward_vanilla_attn(x, mask, rope)
        return self.forward_lambda_attn(x, mask, rope)

    def forward_vanilla_attn(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        broadcasted_fixed_features = self.fixed_features.expand(batch_size, -1, -1)
        x = torch.concat( (broadcasted_fixed_features, x), dim=1 )
        seq_len += self.n_fixed_embeddings

        query = self.query_proj(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)

        keys = self.key_proj(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)

        values = self.value_proj(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)

        # Add rot pos emb to "intent" and "feature"
        if rope is not None:
            query, keys = RotaryPositionEmbedding.apply(
                query, keys, rope
            )

        scores = torch.einsum('bhim,bhjm->bhij', query, keys)
        scores = self.logit_norm(scores, mask)
        # multiply batch, head, i, j by 1,h,1,1 var.
        scores = scores * torch.sqrt(self.norm_variance).view(1, -1, 1, 1)
        scores = scores.masked_fill(~mask[None, None, :, :], float('-inf'))

        # Softmax into attn scores
        attn_weights = F.softmax(scores, dim=-1)
        # Revisit fwd & back to yield the attention result; batch, head, idx, head_dim
        attn =  torch.einsum('bhij,bhjm->bhim', attn_weights, values)
        # print(f"attn_w: \n{attn_weights}")
        # print(f"attn: \n{attn}")

        output = attn.transpose(1, 2).reshape(
            batch_size, seq_len, self.n_embd)
        output = self.out_proj(output)
        output = self.resid_dropout(output)[
            :, self.n_fixed_embeddings:, :
        ]
        
        return output

    def forward_lambda_attn(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        broadcasted_fixed_features = self.fixed_features.expand(batch_size, -1, -1)
        x = torch.concat( (broadcasted_fixed_features, x), dim=1 )
        seq_len += self.n_fixed_embeddings

        # "Q" formulation
        intent_operator = self.intent_operator(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)
        intent_operand = self.intent_operand(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)

        # "Key" formulation
        characteristic_operator = self.characteristic_operator(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)
        characteristic_operand = self.characteristic_operand(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)
        
        # Add rot pos emb to Q&K
        if rope is not None:
            intent_operator, characteristic_operator = RotaryPositionEmbedding.apply(
                intent_operator, characteristic_operator, rope
            )
            intent_operand, characteristic_operand = RotaryPositionEmbedding.apply(
                intent_operand, characteristic_operand, rope
            )
            
        # "Value" formulation
        # Since we have U >= V, muliplying U first is more efficient than multiplying V first.
        features = self.feature_proj(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)
        reshaped_e = x.reshape(
            batch_size, seq_len, self.n_model_split_u, self.n_model_split_v
        )
        reshaped_u = self.op_u.reshape(
            self.n_head, self.head_dim, self.n_model_split_u
        )
        reshaped_v = self.op_v.reshape(
            self.n_head, self.n_model_split_v, self.head_dim
        )
        ue = torch.einsum("hiu,bkuv->bkhiv", reshaped_u, reshaped_e)
        operations = torch.einsum("bkhiv,hvj->bkhij", ue, reshaped_v)
        # batch, head, seq, head_dim, head_dim
        # Each position is sum of head_dim components - variance scale back
        operations = operations.permute(0, 2, 1, 3, 4).contiguous()
        if mask.ndim != 2 or mask.shape != (seq_len, seq_len):
            raise ValueError(
                f"Expected attention mask shape {(seq_len, seq_len)}, "
                f"got {tuple(mask.shape)}"
            )
        # calculated[bhij_] = matmul(operation_i, feature_j)
        # operation: bh[i/j]_ _, feature: bh[i/j]_
        calculated = torch.einsum('bhimn,bhjm->bhijn', operations, features)
        # print(operations.shape)
        # print(features.shape)
        # print(calculated.shape)

        # Compute attention fit scores
        scores_fwd = torch.einsum('bhim,bhjm->bhij', intent_operand, characteristic_operand)
        scores_back = torch.einsum('bhim,bhjm->bhij', intent_operator, characteristic_operator)
        # scores_raw = torch.einsum('bhim,bhjm->bhij', intent, features)
        # Mask out the not-allowed attention positions and i-i double counting
        mask_fwd = mask.clone()
        mask_back = mask.clone()
        mask_fwd.fill_diagonal_(False)
        scores = torch.concat( (scores_fwd, scores_back), dim=-1 )
        mask_total = torch.concat( (mask_fwd, mask_back), dim=-1 )
        scores = self.logit_norm(scores, mask_total)
        # multiply batch, head, i, j by 1,h,1,1 var.
        scores = scores * torch.sqrt(self.norm_variance).view(1, -1, 1, 1)
        scores = scores.masked_fill(~mask_total[None, None, :, :], float('-inf'))
        # print(f"score_fwd: \n{scores_fwd}")
        # print(f"score_b: \n{scores_back}")
        # print(f"score: \n{scores}")

        # Softmax into attn scores
        attn_weights = F.softmax(scores, dim=-1)
        # Revisit fwd & back to yield the attention result; batch, head, idx, head_dim
        attn =  torch.einsum('bhij,bhijm->bhim', attn_weights[:,:,:,:seq_len], calculated) + \
                torch.einsum('bhij,bhjim->bhim', attn_weights[:,:,:,seq_len:], calculated)
        # print(f"attn_w: \n{attn_weights}")
        # print(f"attn: \n{attn}")

        output = attn.transpose(1, 2).reshape(
            batch_size, seq_len, self.n_embd)
        output = self.out_proj(output)
        output = self.resid_dropout(output)[
            :, self.n_fixed_embeddings:, :
        ]
        
        return output


class NormalizedTransformerBlock(nn.Module):
    def __init__(
        self,
        config: NormalizedTransformerConfig,
        n_head: int,
        use_vanilla_attn: bool,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = LambdaAttention(
            config.n_embd,
            n_head,
            config.dropout,
            config.bias,
            use_vanilla_attn,
            config.n_fixed_embeddings,
            config.n_model_split_u,
            config.n_model_split_v,
            config.dtype,
        )
        # TODO - test component performance impact
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        x = x + self.attn(
            self.ln_1(x),
            mask=mask,
            rope=rope,
        )
        x = x + self.mlp(self.ln_2(x))
        return x


class NormalizedTransformer(FusedCrossEntropyLoss, nn.Module):
    def __init__(self, config: NormalizedTransformerConfig):
        super().__init__()
        if config.n_head < 1:
            raise ValueError("n_head must be positive")
        head_dim = config.n_embd // config.n_head
        if config.n_layer < 1:
            raise ValueError("n_layer must be positive")
        if not 0 <= config.feature_selection_layers < config.n_layer:
            raise ValueError(
                "feature_selection_layers must be non-negative and less than n_layer"
            )
        if not config.use_vanilla_attn:
            if config.n_model_split_u < 1 or config.n_model_split_v < 1:
                raise ValueError("Operation split dimensions must be positive")
            if config.n_model_split_u * config.n_model_split_v != config.n_embd:
                raise ValueError(
                    "n_model_split_u * n_model_split_v must equal n_embd, "
                    f"got {config.n_model_split_u} * {config.n_model_split_v} != {config.n_embd}"
                )
            if config.n_model_split_u < config.n_model_split_v:
                print("n_model_split_u is less than n_model_split_v. \
                    Unless this is arbitrary choice by omitting U or V, \
                    consider swapping them for slightly more efficient multiplication."
                )
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if config.feature_selection_heads < 1:
            raise ValueError("feature_selection_heads must be positive")
        if config.feature_selection_backward_distance < 0:
            raise ValueError(
                "feature_selection_backward_distance must be non-negative"
            )
        if config.n_fixed_embeddings < 0:
            raise ValueError("n_fixed_embeddings must be non-negative")
        if config.n_embd % config.feature_selection_heads != 0:
            raise ValueError(
                "n_embd must be divisible by feature_selection_heads"
            )

        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.n_embd, dtype=config.dtype
        )
        self.feature_rotary_embedding = RotaryPositionEmbedding(
            max_seq_len=config.block_size + config.n_fixed_embeddings,
            head_dim=config.n_embd // config.feature_selection_heads,
            dtype=config.dtype,
        )
        self.inference_rotary_embedding = RotaryPositionEmbedding(
            max_seq_len=config.block_size + config.n_fixed_embeddings,
            head_dim=config.n_embd // config.n_head,
            dtype=config.dtype,
        )
        # If not recurrent, don't init it at all - otherwise Muon will think it's in use
        shared_recurrent_block = (
            NormalizedTransformerBlock(
                config, config.n_head, use_vanilla_attn=config.use_vanilla_attn
            )
            if config.recurrent_inference
            else None
        )
        # Always use vanilla on feature selection
        self.transformer = nn.ModuleDict(
            dict(
                feature_selection=nn.ModuleList(
                    [
                        NormalizedTransformerBlock(
                            config,
                            config.feature_selection_heads,
                            use_vanilla_attn=True,
                        )
                        for _ in range(config.feature_selection_layers)
                    ]
                ),
                recurrent=nn.ModuleList(
                    [
                        shared_recurrent_block
                        if shared_recurrent_block is not None
                        else NormalizedTransformerBlock(
                            config,
                            config.n_head,
                            use_vanilla_attn=config.use_vanilla_attn,
                        )
                        for _ in range(
                            config.n_layer - config.feature_selection_layers
                        )
                    ]
                ),
                norm=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False, dtype=config.dtype
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params

    def orthogonal_regularization_loss(self) -> torch.Tensor:
        loss = self.lm_head.weight.new_zeros(())
        for module in self.modules():
            if isinstance(module, LambdaAttention):
                loss = loss + module.orthogonal_regularization_loss()
        return loss

    def _feature_selection_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        aug_to_aug = torch.eye(
            self.config.n_fixed_embeddings,
            self.config.n_fixed_embeddings,
            dtype=torch.bool,
            device=device,
        )
        aug_to_seq = torch.zeros(
            self.config.n_fixed_embeddings,
            seq_len,
            dtype=torch.bool,
            device=device,
        )
        positions = torch.arange(seq_len, device=device)
        dists = positions[:, None] - positions[None, :] # i - j
        seq_to_seq = (dists >= 0) & (
            dists <= self.config.feature_selection_backward_distance
        )
        seq_to_aug = torch.ones(
            seq_len,
            self.config.n_fixed_embeddings,
            dtype=torch.bool,
            device=device,
        )
        concatenated = torch.cat((
            torch.cat((aug_to_aug, aug_to_seq), dim=-1),
            torch.cat((seq_to_aug, seq_to_seq), dim=-1)
        ), dim=0)
        return concatenated

    def _recurrent_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        aug_to_aug = torch.eye(
            self.config.n_fixed_embeddings,
            self.config.n_fixed_embeddings,
            dtype=torch.bool,
            device=device,
        )
        aug_to_seq = torch.zeros(
            self.config.n_fixed_embeddings,
            seq_len,
            dtype=torch.bool,
            device=device,
        )
        seq_to_seq = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )
        seq_to_aug = torch.ones(
            seq_len,
            self.config.n_fixed_embeddings,
            dtype=torch.bool,
            device=device,
        )
        concatenated = torch.cat((
            torch.cat((aug_to_aug, aug_to_seq), dim=-1),
            torch.cat((seq_to_aug, seq_to_seq), dim=-1)
        ), dim=0)
        return concatenated

    def forward(
        self,
        batch: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_accuracy: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len = batch.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {seq_len}, "
                f"block size is only {self.config.block_size}"
            )

        x = self.token_embedding(batch)
        feature_positions = torch.arange(self.config.n_fixed_embeddings + seq_len, device=batch.device).expand(
            batch.size(0), -1
        )
        feature_rope = self.feature_rotary_embedding(feature_positions)
        feature_mask = self._feature_selection_mask(seq_len, batch.device)
        for block in self.transformer.feature_selection:
            x = block(
                x,
                mask=feature_mask,
                rope=feature_rope,
            )

        inference_positions = torch.arange(
            self.config.n_fixed_embeddings + seq_len, device=batch.device
        ).expand(
            batch.size(0), -1
        )
        inference_rope = self.inference_rotary_embedding(inference_positions)
        recurrent_mask = self._recurrent_mask(seq_len, batch.device)
        for block in self.transformer.recurrent:
            kwargs = {
                "mask": recurrent_mask,
            }
            kwargs["rope"] = inference_rope if self.config.rope_enabled_for_inference else None
            x = block(
                x,
                **kwargs
            )
        x = self.transformer.norm(x)

        if targets is None:
            return self.lm_head(x)
        loss = self.cross_entropy_loss(x, self.lm_head, targets)
        if self.config.orthogonal_regularization_factor > 0:
            loss = loss + (
                self.config.orthogonal_regularization_factor
                * self.orthogonal_regularization_loss()
            )
        if return_accuracy:
            with torch.no_grad():
                logits = self.lm_head(x.reshape(-1, x.size(-1)))
                accuracy = logits.argmax(dim=-1).reshape_as(targets) == targets
                accuracy.masked_fill_(targets == -100, False)
            return loss, accuracy
        return loss


class NormalizedModel(ModelBase):
    def __init__(self, config: NormalizedTransformerConfig):
        super().__init__()
        self.config = config
        self.model = NormalizedTransformer(config).to(dtype=config.dtype)
        self._pair_accuracy_counts = None
        self._pair_accuracy_correct = None
        self.optimizer_muon: Optional[torch.optim.Optimizer] = None
        self.optimizer_adamw: Optional[torch.optim.Optimizer] = None

    def setup_fabric(self, fabric: L.Fabric):
        self.fabric = fabric
        if not is_wrapped(self.model):
            self.model = fabric.setup_module(self.model)
            self.fabric.print(self.model)
            self.fabric.print(f"Total number of parameters: {self.get_num_params():,}")
        if self.optimizer_muon is not None and not is_wrapped(self.optimizer_muon):
            if self.optimizer_adamw is None:
                raise RuntimeError("AdamW optimizer was not initialized")
            self.optimizer_muon, self.optimizer_adamw = fabric.setup_optimizers(
                self.optimizer_muon, self.optimizer_adamw
            )
        elif self.optimizer_adamw is not None and not is_wrapped(self.optimizer_adamw):
            self.optimizer_adamw = fabric.setup_optimizers(self.optimizer_adamw)
        if self.optimizer_muon is not None and self.optimizer_adamw is not None:
            self.optimizer = self.optimizer_muon
        else:
            self.optimizer = self.optimizer_adamw

    def compute_loss(
        self,
        batch: torch.Tensor,
        backpropagate: bool,
        no_sync: bool = False,
        loss_div: int = 1,
        **_kwargs,
    ) -> torch.Tensor:
        self._assert_fabric_is_setup()
        del _kwargs
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        # Based on experiment, masking out prediction on prefix helps convergence on the entropy-free targets
        # Unlike raw GPT, it does not need to augment gradient on the exposition
        if not 0 <= self.config.context_length < targets.size(1):
            raise ValueError(
                f"context_length must be in [0, {targets.size(1) - 1}], "
                f"got {self.config.context_length}"
            )
        target_positions = torch.arange(targets.size(1), device=targets.device)
        targets = targets.masked_fill(
            (target_positions < self.config.context_length).unsqueeze(0), -100
        )

        if backpropagate and self._pair_accuracy_counts is None:
            self._pair_accuracy_counts = torch.zeros(
                self.config.block_size, dtype=torch.long, device=batch.device
            )
            self._pair_accuracy_correct = torch.zeros_like(
                self._pair_accuracy_counts
            )

        is_fsdp = isinstance(self.model.module, FSDPModule)
        with self.fabric.no_backward_sync(self.model, no_sync and not is_fsdp):
            if backpropagate:
                loss, accuracy = self.model(
                    inputs, targets=targets, return_accuracy=True
                )
                self._record_pair_accuracy(accuracy, targets)
            else:
                loss = self.model(inputs, targets=targets)
            loss = loss / loss_div
            if backpropagate:
                self.fabric.backward(loss)
        return loss.detach()

    def _record_pair_accuracy(
        self, accuracy: torch.Tensor, targets: torch.Tensor
    ):
        seq_len = targets.size(1)
        self._pair_accuracy_counts[:seq_len].add_((targets != -100).sum(dim=0))
        self._pair_accuracy_correct[:seq_len].add_(accuracy.sum(dim=0))

    def get_custom_metrics(self) -> Dict[str, Any]:
        if self._pair_accuracy_counts is None:
            return {}
        stats = torch.stack(
            [self._pair_accuracy_counts, self._pair_accuracy_correct]
        )
        if self.fabric.world_size > 1:
            stats = self.fabric.all_reduce(stats, reduce_op="sum")
        counts, correct = stats
        observed = torch.nonzero(counts, as_tuple=False).flatten()
        if observed.numel():
            per_position = correct[observed].float() / counts[observed]
            indices = observed.cpu().tolist()
            values = per_position.cpu().tolist()
            metrics = {
                "pair_accuracy_next": per_position.mean().item(),
                "pair_accuracy_prev": None,
                "pair_accuracy_by_index": {
                    f"{index},-1": {"next": accuracy, "prev": None}
                    for index, accuracy in zip(indices, values)
                },
            }
        else:
            metrics = {
                "pair_accuracy_next": None,
                "pair_accuracy_prev": None,
                "pair_accuracy_by_index": {},
            }
        self._pair_accuracy_counts.zero_()
        self._pair_accuracy_correct.zero_()
        return metrics

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        use_fused: Optional[bool] = None,
        use_muon: bool = False,
        muon_momentum: float = 0.95,
    ):
        muon_class = getattr(torch.optim, "Muon", None) if use_muon else None
        if use_muon and muon_class is None:
            raise RuntimeError(
                "Normalized Transformer requires torch.optim.Muon, available in PyTorch 2.9 "
                "and newer."
            )

        named_parameters = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        # print([name for name, _ in named_parameters])
        muon_named_parameters = (
            [
                (name, parameter)
                for name, parameter in named_parameters
                if parameter.ndim == 2
                and not name.endswith("token_embedding.weight")
                and not name.endswith("lm_head.weight")
            ]
            if use_muon
            else []
        )
        # print([name for name, _ in muon_named_parameters])
        muon_parameters = [parameter for _, parameter in muon_named_parameters]
        muon_parameter_ids = {id(parameter) for parameter in muon_parameters}
        adamw_decay = [
            parameter
            for name, parameter in named_parameters
            if parameter.ndim >= 2
            and not name.endswith("token_embedding.weight")
            and not name.endswith("lm_head.weight")
            and id(parameter) not in muon_parameter_ids
        ]
        adamw_no_decay = [
            parameter 
            for name, parameter in named_parameters
            if parameter.ndim < 2
            or name.endswith("token_embedding.weight")
            or name.endswith("lm_head.weight")
            and id(parameter) not in muon_parameter_ids
        ]
        optimizer_parameter_ids = [
            id(parameter)
            for parameter in muon_parameters + adamw_decay + adamw_no_decay
        ]
        trainable_ids = {id(parameter) for _, parameter in named_parameters}
        if (
            len(optimizer_parameter_ids) != len(set(optimizer_parameter_ids))
            or set(optimizer_parameter_ids) != trainable_ids
        ):
            raise RuntimeError(
                "Muon and AdamW parameter groups must cover every trainable "
                "parameter exactly once"
            )

        if muon_parameters:
            assert muon_class is not None
            self.optimizer_muon = muon_class(
                muon_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=muon_momentum,
                adjust_lr_fn="match_rms_adamw",
            )
        else:
            self.optimizer_muon = None
            print("Normalized Transformer is not using Muon optimizer.")
        self.optimizer_adamw = torch.optim.AdamW(
            [
                {"params": adamw_decay, "weight_decay": weight_decay},
                {"params": adamw_no_decay, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=betas,
            eps=1e-8,
            fused=use_fused,
        )
        self.optimizer = self.optimizer_muon if self.optimizer_muon is not None else self.optimizer_adamw

    def optimizer_step(
        self,
        learning_rate: Optional[float] = None,  # If given, use this learning rate
        grad_clip: Optional[float] = None,  # If given, clip gradient norm
    ) -> Optional[torch.Tensor]:
        """
        Performs a single optimization step.
        If gradient clipping is enabled, return the gradient norm before clipping.
        """
        self._assert_fabric_is_setup()

        # Update learning rate
        if learning_rate is not None:
            for optimizer in (self.optimizer_muon, self.optimizer_adamw):
                if optimizer is not None:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = learning_rate

        # Clip gradient
        grad_norm = None
        if grad_clip is not None:
            group_norms = []
            for optimizer in (self.optimizer_muon, self.optimizer_adamw):
                if optimizer is not None:
                    model_grad_norm = self.fabric.clip_gradients(
                        self.model, optimizer, max_norm=grad_clip
                    )
                    group_norms.extend(
                        norm
                        for norm in (model_grad_norm,)
                        if norm is not None
                    )
            if group_norms:
                grad_norm = torch.linalg.vector_norm(torch.stack(group_norms))

        # Step optimizer
        for optimizer in (self.optimizer_muon, self.optimizer_adamw):
            if optimizer is not None:
                optimizer.step()
        for optimizer in (self.optimizer_muon, self.optimizer_adamw):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
        self.training_steps += 1

        # Return gradient norm
        return grad_norm

    def compile(self):
        self._assert_fabric_is_setup(setup=False)
        self.model.compile()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def get_num_params(self, non_embedding: bool = True) -> int:
        return self.model.get_num_params(non_embedding)

    def _get_checkpoint_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "model": self.model,
            "training_steps": self.training_steps,
        }
        if self.optimizer_muon is not None and self.optimizer_adamw is not None:
            state["optimizer_muon"] = self.optimizer_muon
            state["optimizer_adamw"] = self.optimizer_adamw
        elif self.optimizer_adamw is not None:
            state["optimizer"] = self.optimizer_adamw
        return state

    def crop_block_size(self, block_size: int):
        if not 0 < block_size <= self.config.block_size:
            raise ValueError(
                f"block_size must be in [1, {self.config.block_size}], got {block_size}"
            )
        self.config.block_size = block_size
