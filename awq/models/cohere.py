import tqdm
import torch
from typing import List, Tuple
from .base import BaseAWQForCausalLM
from awq.utils.fused_utils import fuse_qkv
from awq.modules.fused.block import LlamaLikeBlock
from awq.modules.fused.model import LlamaLikeModel
from transformers.models.cohere.modeling_cohere import (
    CohereDecoderLayer as OldCohereDecoderLayer,
    CohereForCausalLM as OldCohereForCausalLM,
)
from awq.modules.fused.norm import FasterTransformerRMSNorm


class CohereAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "CohereDecoderLayer"
    max_new_tokens_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model: OldCohereDecoderLayer):
        fuser = CohereFuser(model)
        fuser.fuse_transformer()

    @staticmethod
    def get_model_layers(model: OldCohereForCausalLM):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module: OldCohereDecoderLayer):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldCohereForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(module: OldCohereDecoderLayer, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )

        # linear 1
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )

        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

        return layers


class CohereFuser:
    def __init__(self, model: OldCohereForCausalLM):
        self.model = model

        self.cohere_blocks: List[Tuple[str, OldCohereDecoderLayer]] = [
            (name, module)
            for name, module in self.model.named_modules()
            if "CohereDecoderLayer".lower() in module.__class__.__name__.lower()
        ]

    def fuse_transformer(self):
        blocks = []

        module: OldCohereDecoderLayer
        for module in tqdm.tqdm(self.model.model.layers, desc="Fusing layers..."):
            device = next(iter(module.state_dict().values())).device
            qkv = fuse_qkv(
                module,
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            )
            with torch.no_grad():
                # CohereRMSNorm is different from Llama's in that it multiplies
                # (1 + weight) to the output, instead of just weight.
                module.input_layernorm.weight += 1
                module.post_attention_layernorm.weight += 1
            norm_1 = FasterTransformerRMSNorm(
                module.input_layernorm.weight, module.input_layernorm.eps
            )
            norm_2 = FasterTransformerRMSNorm(
                module.post_attention_layernorm.weight,
                module.post_attention_layernorm.eps,
            )
            blocks.append(
                LlamaLikeBlock(
                    hidden_size=self.model.config.hidden_size,
                    n_heads=self.model.config.num_attention_heads,
                    n_kv_heads=self.model.config.num_key_value_heads,
                    qkv_layer=qkv,
                    o_proj=module.self_attn.o_proj,
                    mlp=module.mlp,
                    norm_1=norm_1,
                    norm_2=norm_2,
                    dev=device,
                    max_seq_len=self.model.config.max_seq_len,
                    rope_theta=self.model.config.rope_theta,
                    head_dim=self.model.config.head_dim,
                )
            )
        
        with torch.no_grad():
            # Normalize Cohere's embedding layer
            self.model.model.embed_tokens.weight *= self.model.config.hidden_size**0.5
        
        self.model.model = LlamaLikeModel(
            self.model.config.vocab_size,
            blocks,
            self.model.model.embed_tokens,
            self.model.model.norm,
        )
        setattr(self.model.model, "blocks", self.model.model.blocks)
