"""
Feature-selection and recurrent-inference language model.

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
)


@dataclass
class FeatureRNNConfig:
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
    n_fixed_feature_embeddings: int = 16
    n_model_split_u: int = 64
    n_model_split_v: int = 4
    omit_u_matrix: bool = False
    omit_v_matrix: bool = False
    orthogonal_regularization_factor: float = 0.0

class MLP(nn.Module):
    def __init__(self, config: FeatureRNNConfig):
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

class FeatureAttention(nn.Module):
    """Highly optimized Attention with decomposed U and V operation matrices and intent-feature casting."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        bias: bool,
        use_vanilla_attn: bool,
        n_fixed_feature_embeddings: int,
        n_model_split_u: int,
        n_model_split_v: int,
        omit_u_matrix: bool,
        omit_v_matrix: bool,
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
        self.n_fixed_feature_embeddings = n_fixed_feature_embeddings
        self.n_model_split_u = n_model_split_u
        self.n_model_split_v = n_model_split_v
        self.omit_u_matrix = omit_u_matrix
        self.omit_v_matrix = omit_v_matrix
        self.dtype = dtype
        self.use_vanilla_attn = use_vanilla_attn

        self.fixed_features = nn.Parameter(
            torch.empty(1, n_fixed_feature_embeddings, n_embd, dtype=dtype)
        )

        # self.intent = nn.Linear(n_embd, n_embd, bias=bias, dtype=dtype)
        self.intent_operator = nn.Linear(n_embd, n_embd, bias=bias, dtype=dtype)
        self.feature = nn.Linear(n_embd, n_embd, bias=bias, dtype=dtype)
        self.value_proj = None
        self.op_u = None
        self.op_v = None
        if self.use_vanilla_attn:
            self.value_proj = nn.Linear(n_embd, n_embd, bias=bias, dtype=dtype)
        else:
            # Operand query only applies for lambda attn
            # When the current embedding is testified as operand of previous embedding
            self.intent_operand = nn.Linear(n_embd, n_embd, bias=bias, dtype=dtype)
            # Need to declare them as 2D to allow Muon to pick them up as params
            self.op_u = (
                nn.Parameter(
                    torch.empty(n_head * self.head_dim, n_model_split_u, dtype=dtype)
                )
                if not omit_u_matrix
                else None
            )
            self.op_v = (
                nn.Parameter(
                    torch.empty(n_head * n_model_split_v, self.head_dim, dtype=dtype)
                )
                if not omit_v_matrix
                else None
            )
        # TODO - test component performance impact
        self.logit_norm = nn.InstanceNorm2d(num_features=self.n_head, affine=False)
        self.output_proj = nn.Linear(n_embd, n_embd, bias=bias, dtype=dtype)
        self.dropout = dropout
        self.resid_dropout = (
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        nn.init.normal_(self.fixed_features, mean=0.0, std=1.0)
        if self.op_u is not None:
            nn.init.normal_(self.op_u, mean=0.0, std=1.0 / self.sqrt_head_dim)
        if self.op_v is not None:
            nn.init.normal_(self.op_v, mean=0.0, std=1.0 / self.sqrt_head_dim)

    def orthogonal_regularization_loss(self) -> torch.Tensor:
        loss = self.fixed_features.new_zeros(())
        if self.op_u is not None:
            u = self.op_u.reshape(
                self.n_head, self.head_dim, self.n_model_split_u
            ).float()
            identity_u = torch.eye(
                self.n_model_split_u, device=u.device, dtype=u.dtype
            )
            loss = loss + (u.transpose(-1, -2) @ u - identity_u).square().mean()
        if self.op_v is not None:
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
        seq_len += self.n_fixed_feature_embeddings

        # "Key" in both vanilla / lambda attention
        keys = self.feature(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)

        # "Value" formulation
        values = self.value_proj(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)

        # "Q" formulation
        query = self.intent_operator(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)
        # Add rot pos emb to "intent" and "feature"
        if rope is not None:
            query, keys = RotaryPositionEmbedding.apply(
                query, keys, rope
            )

        scores = torch.einsum('bhim,bhjm->bhij', query, keys)
        scores = self.logit_norm(scores)
        scores = scores.masked_fill(~mask[None, None, :, :], float('-inf'))

        # Softmax into attn scores
        attn_weights = F.softmax(scores, dim=-1)
        # Revisit fwd & back to yield the attention result; batch, head, idx, head_dim
        attn =  torch.einsum('bhij,bhjm->bhim', attn_weights, values)
        # print(f"attn_w: \n{attn_weights}")
        # print(f"attn: \n{attn}")

        output = attn.transpose(1, 2).reshape(
            batch_size, seq_len, self.n_embd)
        output = self.output_proj(output)
        output = self.resid_dropout(output)[
            :, self.n_fixed_feature_embeddings:, :
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
        seq_len += self.n_fixed_feature_embeddings

        # "Key" in both vanilla / lambda attention
        features = self.feature(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)

        # "Value" formulation
        # Since we have U >= V, muliplying U first is more efficient than multiplying V first.
        reshaped_e = x.reshape(
            batch_size, seq_len, self.n_model_split_u, self.n_model_split_v
        )
        # If ommitting u/v, they will be effectively identity matrix.
        if self.op_u is None:
            ue = reshaped_e.unsqueeze(2).expand(
                -1, -1, self.n_head, -1, -1
            )
        else:
            reshaped_u = self.op_u.reshape(
                self.n_head, self.head_dim, self.n_model_split_u
            )
            ue = torch.einsum("hiu,bkuv->bkhiv", reshaped_u, reshaped_e)
        if self.op_v is None:
            operations = ue
        else:
            reshaped_v = self.op_v.reshape(
                self.n_head, self.n_model_split_v, self.head_dim
            )
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

        # "Q" formulation
        intent_operator = self.intent_operator(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)
        intent_operand = self.intent_operand(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)
        # Add rot pos emb to "intent" and "feature"
        if rope is not None:
            combined_intent = torch.concat((intent_operator, intent_operand), dim=1)
            combined_intent, features = RotaryPositionEmbedding.apply(
                combined_intent, features, rope
            )
            intent_operator = combined_intent[:, :self.n_head, :, :]
            intent_operand = combined_intent[:, self.n_head:, :, :]

        # Sadly, our attn depends on both i and j - flash attention not applicable...
        scores_fwd = torch.einsum('bhim,bhjm->bhij', intent_operand, features)
        scores_back = torch.einsum('bhim,bhjm->bhij', intent_operator, features)
        # scores_raw = torch.einsum('bhim,bhjm->bhij', intent, features)
        # Mask out the not-allowed attention positions and i-i double counting
        mask_fwd = mask.clone()
        mask_back = mask.clone()
        mask_fwd.fill_diagonal_(False)
        scores_fwd = self.logit_norm(scores_fwd)
        scores_back = self.logit_norm(scores_back)
        scores_fwd = scores_fwd.masked_fill(~mask_fwd[None, None, :, :], float('-inf'))
        scores_back = scores_back.masked_fill(~mask_back[None, None, :, :], float('-inf'))
        scores = torch.concat((scores_fwd, scores_back), dim=-1)
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
        output = self.output_proj(output)
        output = self.resid_dropout(output)[
            :, self.n_fixed_feature_embeddings:, :
        ]
        
        return output


class FeatureRNNBlock(nn.Module):
    def __init__(
        self,
        config: FeatureRNNConfig,
        n_head: int,
        use_vanilla_attn: bool,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = FeatureAttention(
            config.n_embd,
            n_head,
            config.dropout,
            config.bias,
            use_vanilla_attn,
            config.n_fixed_feature_embeddings,
            config.n_model_split_u,
            config.n_model_split_v,
            config.omit_u_matrix,
            config.omit_v_matrix,
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


class FeatureRNNTransformer(FusedCrossEntropyLoss, nn.Module):
    def __init__(self, config: FeatureRNNConfig):
        super().__init__()
        if config.n_layer < 1:
            raise ValueError("n_layer must be positive")
        if not 0 <= config.feature_selection_layers < config.n_layer:
            raise ValueError(
                "feature_selection_layers must be non-negative and less than n_layer"
            )
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
        if config.n_head < 1:
            raise ValueError("n_head must be positive")
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        head_dim = config.n_embd // config.n_head
        if config.omit_u_matrix and config.n_model_split_u != head_dim:
            raise ValueError(
                "n_model_split_u must be equal to attn head dimension when omit_u_matrix is on"
            )
        if config.omit_v_matrix and config.n_model_split_v != head_dim:
            raise ValueError(
                "n_model_split_v must be equal to attn head dimension when omit_v_matrix is on"
            )
        if config.feature_selection_heads < 1:
            raise ValueError("feature_selection_heads must be positive")
        if config.feature_selection_backward_distance < 0:
            raise ValueError(
                "feature_selection_backward_distance must be non-negative"
            )
        if config.n_fixed_feature_embeddings < 0:
            raise ValueError("n_fixed_feature_embeddings must be non-negative")
        if config.n_embd % config.feature_selection_heads != 0:
            raise ValueError(
                "n_embd must be divisible by feature_selection_heads"
            )

        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.n_embd, dtype=config.dtype
        )
        self.feature_rotary_embedding = RotaryPositionEmbedding(
            max_seq_len=config.block_size,
            head_dim=config.n_embd // config.feature_selection_heads,
            dtype=config.dtype,
        )
        self.inference_rotary_embedding = RotaryPositionEmbedding(
            max_seq_len=config.block_size + config.n_fixed_feature_embeddings,
            head_dim=config.n_embd // config.n_head,
            dtype=config.dtype,
        )
        # If not recurrent, don't init it at all - otherwise Muon will think it's in use
        shared_recurrent_block = (
            FeatureRNNBlock(
                config, config.n_head, use_vanilla_attn=config.use_vanilla_attn
            )
            if config.recurrent_inference
            else None
        )
        self.transformer = nn.ModuleDict(
            dict(
                feature_selection=nn.ModuleList(
                    [
                        FeatureRNNBlock(
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
                        else FeatureRNNBlock(
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
            if isinstance(module, FeatureAttention):
                loss = loss + module.orthogonal_regularization_loss()
        return loss

    def _feature_selection_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        backward_distance = positions[:, None] - positions[None, :]
        return (backward_distance >= 0) & (
            backward_distance <= self.config.feature_selection_backward_distance
        )

    def _recurrent_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        aug_to_aug = torch.eye(
            self.config.n_fixed_feature_embeddings,
            self.config.n_fixed_feature_embeddings,
            dtype=torch.bool,
            device=device,
        )
        aug_to_seq = torch.zeros(
            self.config.n_fixed_feature_embeddings,
            seq_len,
            dtype=torch.bool,
            device=device,
        )
        seq_to_seq = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )
        seq_to_aug = torch.ones(
            seq_len,
            self.config.n_fixed_feature_embeddings,
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
        feature_positions = torch.arange(seq_len, device=batch.device).expand(
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
            self.config.n_fixed_feature_embeddings + seq_len, device=batch.device
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


class FeatureRNN(ModelBase):
    def __init__(self, config: FeatureRNNConfig):
        super().__init__()
        self.config = config
        self.model = FeatureRNNTransformer(config).to(dtype=config.dtype)
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
        # Unlike baseline GPT models, feature attention only sees marginal gains by including prefix
        # Isn't woth the extra effort to predict entropy-heavy prefix
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
                "FeatureRNN requires torch.optim.Muon, available in PyTorch 2.9 "
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
            print("FeatureRNN is not using Muon optimizer.")
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
