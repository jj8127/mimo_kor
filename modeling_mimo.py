from typing import Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import (Qwen2Attention,
                                                      Qwen2ForCausalLM,
                                                      Qwen2MLP, Qwen2Model,
                                                      Qwen2RMSNorm)

from .configuration_mimo import MiMoConfig

# MTP Layer 정의
class MiMoMTPLayers(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Qwen2RMSNorm과 Qwen2Attention, Qwen2MLP를 사용하여 MTP Layer를 구성
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.token_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # input_proj는 input embedding 과 hidden state를 Concat하여 projection
        # 여러 단게의 LayerNorm을 거쳐 안정화된 학습 유도
        self.input_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.final_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen2Attention(config, layer_idx=0)
        self.mlp = Qwen2MLP(config)

    # 입력 정규화 => concat => projection => self-attention => MLP => 정규화의 전형적인 transformer 구조
    # forward 함수는 입력을 받아서 MTP Layer를 통과시킴
    
    # Qwen2 스타일로 layernorm => attention => residual => layerNorm => MLP 순으로 처리
    # 마지막에 final layernorm을 수행한 후 출력
    def forward(self, input_embeds,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values: Optional[Cache]=None,
                    output_attentions: Optional[bool]=False,
                    use_cache: Optional[bool]=False,
                    position_embedding: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    cache_position=None,
                    **kwargs):
        input_embeds = self.token_layernorm(input_embeds)
        previous_hidden_states = self.hidden_layernorm(hidden_states)
        hidden_states = self.input_proj(torch.cat([previous_hidden_states, input_embeds], dim=-1))
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states,
                                       attention_mask=attention_mask,
                                       position_ids=position_ids,
                                       past_key_values=past_key_values,
                                       output_attentions=output_attentions,
                                       use_cache=use_cache,
                                       cache_position=cache_position,
                                       position_embedding=position_embedding,
                                       **kwargs)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states

# MiMoModel 정의
# MiMoModel은 Qwen2Model을 상속받아 MTP Layer를 추가한 모델
class MiMoModel(Qwen2Model):
    config_class = MiMoConfig

    def __init__(self, config: MiMoConfig):
        super().__init__(config)
        # inference 시 speculative decoding을 위해 여러 MTP 레이어를 사용하는 구조와 맞닿아 있음
        # MTP Layer를 여러 개 쌓아놓고, 각 레이어를 순차적으로 통과시키는 구조
        self.mtp_layers = nn.ModuleList([MiMoMTPLayers(config) for _ in range(config.num_nextn_predict_layers)])

# 최종 언어 생성 모델
# Qwen2ForCausalLM을 기반으로, 내부 모델을 MiMoModel로 변경
class MiMoForCausalLM(Qwen2ForCausalLM):
    config_class = MiMoConfig
    def __init__(self, config: MiMoConfig):
        super(Qwen2ForCausalLM, self).__init__(config)
        # 최종적으로 Transformer 출력 => lm_head => logits => token 확률 분포
        self.model = MiMoModel(config)
        self.vocab_size = config.vocab_size
        # lm_head는 언어 모델링을 위한 최종 레이어
        # hidden state를 vocab size로 projection 하여 각 단어의 확률을 계산
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()
