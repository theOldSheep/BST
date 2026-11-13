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

# Debug flag
FEATURE_RNN_DEBUG_GRADIENTS = False


@dataclass
class FeatureRNNConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False
    context_length: int = 0
    use_fused: bool = False
    feature_selection_layers: int = 4
    feature_selection_heads: int = 4
    feature_selection_backward_distance: int = 7
    muon_momentum: float = 0.95
    rope_enabled_for_inference: bool = True
    recurrent_inference: bool = False
    n_fixed_feature_embeddings: int = 16
    n_model_split_u: int = 64
    n_model_split_v: int = 4

class VanillaSelfAttention(nn.Module):
    def __init__(self, config: FeatureRNNConfig, n_head: Optional[int] = None):
        super().__init__()
        self.n_head = n_head if n_head else config.n_head
        assert config.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        self.resid_dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # PyTorch SDPA selects an available fused backend when supported.
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # input should have shape (batch_size, seq_len, n_embd)
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # convert to shape (B, n_head, T, d_head)
        d_head = C // self.n_head
        k = k.view(B, T, self.n_head, d_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, d_head).transpose(1, 2)

        # apply rotary position embedding to q and k
        if rope is not None:
            q, k = RotaryPositionEmbedding.apply(q, k, rope)

        # causal self-attention; Self-attend: (B, Nh, T, Dh) x (B, Nh, Dh, T) -> (B, Nh, T, T)
        if self.flash:
            # Use PyTorch's fused scaled-dot-product attention backend.
            if mask is not None:
                mask = mask.view(1, 1, T, T)
            y = torch.nn.functional.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=(mask is None),
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            if mask is not None:
                att.masked_fill_(mask.logical_not(), float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).reshape(
            B, T, C
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config: FeatureRNNConfig):
        super().__init__()
        hidden_dim = 8 * config.n_embd / 3
        hidden_dim = 128 * round(hidden_dim / 128)
        self.up = SwiGLU(config.n_embd, hidden_dim, bias=config.bias)
        self.down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.down(x)
        x = self.dropout(x)
        return x

class VanillaAttentionBlock(nn.Module):
    def __init__(self, config: FeatureRNNConfig, n_head: Optional[int] = None):
        super().__init__()
        self.ln_1 = LayerNorm(
            config.n_embd,
            bias=config.bias,
        )
        self.attn = VanillaSelfAttention(config, n_head=n_head)
        self.ln_2 = LayerNorm(
            config.n_embd,
            bias=config.bias,
        )
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), mask=mask, rope=rope)
        x = x + self.mlp(self.ln_2(x))
        return x

class FeatureAttention(nn.Module):
    """Highly optimized Attention with decomposed U and V operation matrices and intent-feature casting."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        bias: bool,
        n_fixed_feature_embeddings: int,
        n_model_split_u: int,
        n_model_split_v: int,
    ):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by the attention head count")

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd
        self.n_fixed_feature_embeddings = n_fixed_feature_embeddings
        self.n_model_split_u = n_model_split_u
        self.n_model_split_v = n_model_split_v

        self.fixed_features = nn.Parameter(
            torch.empty(1, n_fixed_feature_embeddings, n_embd)
        )

        self.intent = nn.Linear(n_embd, n_embd, bias=bias)
        self.feature = nn.Linear(n_embd, n_embd, bias=bias)
        # Need to declare them as 2D to allow Muon to pick them up as params
        self.op_u = nn.Parameter(
            torch.empty(n_head * self.head_dim, n_model_split_u)
        )
        self.op_v = nn.Parameter(
            torch.empty(n_head * n_model_split_v, self.head_dim)
        )
        self.output_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.dropout = dropout
        self.resid_dropout = (
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        nn.init.normal_(self.fixed_features, mean=0.0, std=0.02)
        nn.init.normal_(self.op_u, mean=0.0, std=0.02)
        nn.init.normal_(self.op_v, mean=0.0, std=0.02)
        self.debug_name = "feature_attention"
        self._debug_gradient_reported = False

    def _debug_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        stats_values: Optional[torch.Tensor] = None,
    ):
        values = (tensor if stats_values is None else stats_values).detach().float()
        print(
            f"[FeatureRNN debug] {self.debug_name}.{name}: "
            f"shape={tuple(values.shape)} finite={torch.isfinite(values).all().item()} "
            f"abs_max={values.abs().max().item()} "
            f"rms={values.square().mean().sqrt().item()}"
        )
        if tensor.requires_grad:
            def report_gradient(gradient: torch.Tensor) -> torch.Tensor:
                if gradient is not None:
                    values = gradient.detach().float()
                    print(
                        f"[FeatureRNN debug] {self.debug_name}.{name}.grad: "
                        f"finite={torch.isfinite(values).all().item()} "
                        f"abs_max={values.abs().max().item():.4e} "
                        f"rms={values.square().mean().sqrt().item():.4e}"
                    )
                else:
                    print(f"[FeatureRNN debug] {self.debug_name}.{name}.grad: None")
                return gradient

            tensor.register_hook(report_gradient)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        broadcasted_fixed_features = self.fixed_features.repeat((batch_size, 1, 1))
        x = torch.concat( (broadcasted_fixed_features, x), dim=1 )
        seq_len += self.n_fixed_feature_embeddings

        intent = self.intent(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)
        features = self.feature(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)

        if rope is not None:
            intent, features = RotaryPositionEmbedding.apply(
                intent, features, rope
            )

        # Since we have U >= V, muliplying U first is more efficient than multiplying V first.
        reshaped_e = x.reshape(
            batch_size, seq_len, self.n_model_split_u, self.n_model_split_v
        )
        reshaped_u = self.op_u.reshape(
            self.n_head, self.head_dim, self.n_model_split_u
        )
        reshaped_v = self.op_v.reshape(
            self.n_head, self.n_model_split_v, self.head_dim
        )
        ue = (
            torch.einsum("hiu,bkuv->bkhiv", reshaped_u, reshaped_e)
        )
        operations = torch.einsum("bkhiv,hvj->bkhij", ue, reshaped_v)
        # batch, head, seq, head_dim, head_dim
        operations = operations.permute(0, 2, 1, 3, 4).contiguous()

        if mask.ndim != 2 or mask.shape != (seq_len, seq_len):
            raise ValueError(
                f"Expected attention mask shape {(seq_len, seq_len)}, "
                f"got {tuple(mask.shape)}"
            )

        # calculated[bhij_] = matmul(operation_i, feature_j)
        # operation: bh[i/j]_ _, feature: bh[i/j]_
        calculated = torch.einsum('bhimn,bhjn->bhijm', operations, features)
        # print(operations.shape)
        # print(features.shape)
        # print(calculated.shape)

        # Sadly, our attn depends on both i and j - flash attention not applicable...
        scale = math.sqrt(self.head_dim)
        scores_fwd = torch.einsum('bhim,bhijm->bhij', intent, calculated) / scale
        scores_back = torch.einsum('bhim,bhjim->bhij', intent, calculated) / scale
        # Mask out the not-allowed attention positions
        scores_fwd = scores_fwd.masked_fill(~mask[None, None, :, :], float('-inf'))
        scores_back = scores_back.masked_fill(~mask[None, None, :, :], float('-inf'))
        scores = torch.concat((scores_fwd, scores_back), dim=-1)
        # If row fully masked out, need to fill with zero score to prevent -inf mess up grad
        has_visible_keys = mask.any(dim=-1)[None, None, :, None]
        scores = torch.where(has_visible_keys, scores, torch.zeros_like(scores))
        # print(f"score_fwd: \n{scores_fwd}")
        # print(f"score_b: \n{scores_back}")
        # print(f"score: \n{scores}")
        # Softmax and sum over "j" overall
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights.masked_fill(~has_visible_keys, 0.0)
        # Revisit fwd & back to yield the attention result; batch, head, idx, head_dim
        attn =  torch.einsum('bhij,bhijm->bhim', attn_weights[:,:,:,:seq_len], calculated) + \
                torch.einsum('bhij,bhjim->bhim', attn_weights[:,:,:,seq_len:], calculated)
        # print(f"attn_w: \n{attn_weights}")
        # print(f"attn: \n{attn}")

        output = attn.transpose(1, 2).reshape(
            batch_size, seq_len, self.n_embd)
        output = self.resid_dropout(self.output_proj(output))[
            :, self.n_fixed_feature_embeddings:, :
        ]
        if (
            FEATURE_RNN_DEBUG_GRADIENTS
            and self.training
            and not self._debug_gradient_reported
        ):
            for name, tensor in (
                ("intent", intent),
                ("features", features),
                ("operations", operations),
                ("output", output),
                ("u", self.op_u),
                ("v", self.op_v),
            ):
                self._debug_tensor(name, tensor)
            self._debug_gradient_reported = True
        return output


class FeatureRNNBlock(nn.Module):
    def __init__(
        self,
        config: FeatureRNNConfig,
        n_head: int
    ):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = FeatureAttention(
            config.n_embd,
            n_head,
            config.dropout,
            config.bias,
            config.n_fixed_feature_embeddings,
            config.n_model_split_u,
            config.n_model_split_v,
        )
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
        return x + self.mlp(self.ln_2(x))


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
            raise ValueError(
                "n_model_split_u must be no less than n_model_split_v for efficient multiplication, "
                "swap them to pass this validation."
            )
        if config.feature_selection_heads < 0:
            raise ValueError("feature_selection_heads must be non-negative")
        if config.feature_selection_backward_distance < 0:
            raise ValueError(
                "feature_selection_backward_distance must be non-negative"
            )
        if config.n_fixed_feature_embeddings < 0:
            raise ValueError("n_fixed_feature_embeddings must be non-negative")
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if config.n_embd % config.feature_selection_heads != 0:
            raise ValueError(
                "n_embd must be divisible by feature_selection_heads"
            )

        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.feature_rotary_embedding = RotaryPositionEmbedding(
            max_seq_len=config.block_size,
            head_dim=config.n_embd // config.feature_selection_heads,
        )
        self.inference_rotary_embedding = RotaryPositionEmbedding(
            max_seq_len=config.block_size + config.n_fixed_feature_embeddings,
            head_dim=config.n_embd // config.n_head,
        )
        # If not recurrent, don't init it at all - otherwise Muon will think it's in use
        shared_recurrent_block = (
            FeatureRNNBlock(config, config.n_head)
            if config.recurrent_inference
            else None
        )
        self.transformer = nn.ModuleDict(
            dict(
                feature_selection=nn.ModuleList(
                    [
                        VanillaAttentionBlock(
                            config, config.feature_selection_heads
                        )
                        for _ in range(config.feature_selection_layers)
                    ]
                ),
                recurrent=nn.ModuleList(
                    [
                        shared_recurrent_block
                        if shared_recurrent_block is not None
                        else FeatureRNNBlock(config, config.n_head)
                        for _ in range(
                            config.n_layer - config.feature_selection_layers
                        )
                    ]
                ),
                norm=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        for name, module in self.named_modules():
            if isinstance(module, FeatureAttention):
                module.debug_name = name

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
        aug_to_aug = torch.zeros(
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
        if return_accuracy:
            loss = self.cross_entropy_loss(x, self.lm_head, targets)
            with torch.no_grad():
                logits = self.lm_head(x.reshape(-1, x.size(-1)))
                accuracy = logits.argmax(dim=-1).reshape_as(targets) == targets
                accuracy.masked_fill_(targets == -100, False)
            return loss, accuracy
        return self.cross_entropy_loss(x, self.lm_head, targets)


class FeatureRNN(ModelBase):
    def __init__(self, config: FeatureRNNConfig):
        super().__init__()
        self.config = config
        self.model = FeatureRNNTransformer(config)
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
        if self.optimizer_muon is not None and self.optimizer_adamw is not None:
            self.optimizer = self.optimizer_muon

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
        # Based on experiment, masking out prediction on prefix hinders convergence
        # Although edge token prediction introduces token-wise entropy from randomness and permutation
        # Prediction on edges seems to augment gradient & allow model to learn the question better
        # if not 0 <= self.config.context_length < targets.size(1):
        #     raise ValueError(
        #         f"context_length must be in [0, {targets.size(1) - 1}], "
        #         f"got {self.config.context_length}"
        #     )
        # target_positions = torch.arange(targets.size(1), device=targets.device)
        # targets = targets.masked_fill(
        #     (target_positions < self.config.context_length).unsqueeze(0), -100
        # )

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
    ):
        muon_class = getattr(torch.optim, "Muon", None)
        if muon_class is None:
            raise RuntimeError(
                "FeatureRNN requires torch.optim.Muon, available in PyTorch 2.9 "
                "and newer."
            )

        named_parameters = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        print(f"Configuring optimizers for {len(named_parameters)} trainable parameters")
        muon_named_parameters = [
            (name, parameter)
            for name, parameter in named_parameters
            if parameter.ndim == 2
            and not name.endswith("token_embedding.weight")
            and not name.endswith("lm_head.weight")
        ]
        muon_parameters = [parameter for _, parameter in muon_named_parameters]
        print(f"Muon Params: {[name for name, _ in muon_named_parameters]}")
        adamw_decay_parameters = [
            (name, parameter)
            for name, parameter in named_parameters
            if parameter.ndim >= 2
            and (
                parameter.ndim != 2
                or name.endswith("token_embedding.weight")
                or name.endswith("lm_head.weight")
            )
        ]
        adamw_decay = [parameter for _, parameter in adamw_decay_parameters]
        print(f"Adamw Decay Params: {[name for name, _ in adamw_decay_parameters]}")
        adamw_no_decay_parameters = [
            (name, parameter) for name, parameter in named_parameters if parameter.ndim < 2
        ]
        adamw_no_decay = [parameter for _, parameter in adamw_no_decay_parameters]
        print(f"Adamw No Decay Params: {[name for name, _ in adamw_no_decay_parameters]}")
        # Sanity checks
        trainable_ids = {id(parameter) for _, parameter in named_parameters}
        muon_parameter_ids = {id(parameter) for parameter in muon_parameters}
        optimizer_parameter_ids = [
            id(parameter)
            for parameter in muon_parameters + adamw_decay + adamw_no_decay
        ]
        if (
            len(optimizer_parameter_ids) != len(set(optimizer_parameter_ids))
            or set(optimizer_parameter_ids) != trainable_ids
        ):
            raise RuntimeError(
                "Muon and AdamW parameter groups must cover every trainable "
                "parameter exactly once"
            )
        if not muon_parameters:
            raise ValueError("FeatureRNN has no 2D hidden-layer parameters for Muon")
        print(f"Muon parameter names: {[name for name, _ in muon_named_parameters]}")
        print(
            "AdamW-only parameter names: "
            f"{[name for name, parameter in named_parameters if id(parameter) not in muon_parameter_ids]}"
        )

        self.optimizer_muon = muon_class(
            muon_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=self.config.muon_momentum,
            adjust_lr_fn="match_rms_adamw",
        )
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
        self.optimizer = self.optimizer_muon

    def optimizer_step(
        self,
        learning_rate: Optional[float] = None,
        grad_clip: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        self._assert_fabric_is_setup()
        if self.optimizer_muon is None or self.optimizer_adamw is None:
            raise RuntimeError("Optimizers must be set up before calling this function")
        optimizers = (self.optimizer_muon, self.optimizer_adamw)
        if learning_rate is not None:
            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = learning_rate
        grad_norm = None
        if grad_clip is not None:
            group_norms = []
            for optimizer in optimizers:
                group_norm = self.fabric.clip_gradients(
                    self.model, optimizer, max_norm=grad_clip
                )
                if group_norm is not None:
                    group_norms.append(group_norm)
            if group_norms:
                grad_norm = torch.linalg.vector_norm(torch.stack(group_norms))
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        self.training_steps += 1
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
        return state

    def crop_block_size(self, block_size: int):
        if not 0 < block_size <= self.config.block_size:
            raise ValueError(
                f"block_size must be in [1, {self.config.block_size}], got {block_size}"
            )
        self.config.block_size = block_size
