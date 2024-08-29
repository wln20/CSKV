"""
PyTorch LLaMA Attention model from llama: 
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
"""
import math
import logging

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.cache_utils import Cache, StaticCache, DynamicCache
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    is_flash_attn_greater_or_equal_2_10,
    apply_rotary_pos_emb,
    repeat_kv,
    _get_unpad_data,
    add_start_docstrings_to_model_forward,
    LLAMA_INPUTS_DOCSTRING,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)

from transformers.utils import is_flash_attn_2_available
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

from cskv_src.utils.quant_utils import KVQuantizer, KVQuantizerChannel
from cskv_src.utils.cache_utils import DynamicCacheWithWindow

# >>>>>>>>>>>>>>>>>>>>>>> generation method for Window-Based KV Cache >>>>>>>>>>>>>>>>>>>>>>>>
# make sure the generation keeps track of the correct token numbers
def prepare_inputs_for_generation_llama(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    past_length = 0
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            if self.use_window:
                cache_length = past_length = past_key_values[0][0].shape[2] + past_key_values[0][4].shape[2]    # compressed_quant + origin
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    if self.generation_config.cache_implementation == "static":
        # generation with static cache
        cache_position = kwargs.get("cache_position", None)
        if cache_position is None:
            past_length = 0
        else:
            past_length = cache_position[-1] + 1
        input_ids = input_ids[:, past_length:]
        position_ids = position_ids[:, past_length:]
    # TODO @gante we should only keep a `cache_position` in generate, and do +=1.
    # same goes for position ids. Could also help with continued generation.
    cache_position = torch.arange(past_length, past_length + position_ids.shape[-1], device=position_ids.device)

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
        # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
        # TODO: use `next_tokens` directly instead.
        model_inputs = {"input_ids": input_ids.contiguous()}

    model_inputs.update(
        {
            "position_ids": position_ids.contiguous(),
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs
# truncate the input length to be divisible by q_window_size
def forward_llama(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
):
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # ===========================
    if self.is_training and self.use_window:    # when using window, must let the seq len to be multiple of q_window_size
        ori_len = input_ids.shape[1]
        if ori_len % self.q_window_size != 0:
            trunc_len = ori_len - ori_len % self.q_window_size
            if trunc_len == 0:  # the original sequence is shorter than window size
                logging.warn(f"Sequence length ({ori_len}) is not divisible by q_window_size ({self.q_window_size}), padding it to {self.q_window_size}.")
                input_ids = torch.cat((input_ids, torch.ones(input_ids.shape[0], self.q_window_size - ori_len).to(input_ids.device).to(input_ids.dtype)), dim=1)
                if labels is not None:
                    labels = torch.cat((labels, torch.ones(labels.shape[0], self.q_window_size - ori_len).to(labels.device).to(labels.dtype)), dim=1)
            else:
                logging.warn(f"Sequence length ({ori_len}) is not divisible by q_window_size ({self.q_window_size}), truncating it to {trunc_len}.")
                input_ids = input_ids[:, : trunc_len]
                if labels is not None:
                    labels = labels[:, : trunc_len]
            
    # ===========================

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels).to(input_ids.device)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits.to(input_ids.device),
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    
# change the type of DynamicCache
def forward_llama_model(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logging.warn(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    past_seen_tokens = 0
    if use_cache:  # kept for BC (cache positions)
        if not isinstance(past_key_values, StaticCache):
            if self.use_window:
                past_key_values = DynamicCacheWithWindow.from_legacy_cache(past_key_values, self.q_window_size)
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_seen_tokens = past_key_values.get_seq_length()

    if cache_position is None:
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
        )
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
    

# >>>>>>>>>>>>>>>>>>>>>> Channel Reduction for SVD + Parallel >>>>>>>>>>>>>>>>>>>>>>>>
class LlamaAttentionForCSKV(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None, args=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            print(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        # self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

        self.k_compressed_dim_per_head = args.k_compressed_dim
        self.v_compressed_dim_per_head = args.v_compressed_dim
        self.k_compressed_dim = self.k_compressed_dim_per_head * self.num_key_value_heads # Note: here k_compressed_dim should be num_heads*head_dim, eg. 32*64
        self.v_compressed_dim = self.v_compressed_dim_per_head * self.num_key_value_heads

        assert not config.attention_bias, "Bias is not supported in SVD initialization"
        self.k_proj_a = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim + self.k_compressed_dim, bias=False)    # cat(k_proj, U_k\cdot\sqrt{\Sigma_k})
        self.k_proj_b = nn.Linear(self.k_compressed_dim, self.num_key_value_heads * self.head_dim, bias=False)    # \sqrt{\Sigma_k}\cdot V_k^T
        self.v_proj_a = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim + self.v_compressed_dim, bias=False)
        self.v_proj_b = nn.Linear(self.v_compressed_dim, self.num_key_value_heads * self.head_dim, bias=False)

        # quantization
        # window based
        self.use_window = args.use_window    
        self.q_window_size = args.q_window_size
        if self.use_window:  # according to KIVI, we only use per-channel quant for Key Cache, while keeping per-token quant for Value Cache
            self.quantizer_k = KVQuantizerChannel
        else:
            self.quantizer_k = KVQuantizer
        self.quantizer_v = KVQuantizer 
        self.k_bits = args.k_bits
        self.v_bits = args.v_bits
        
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        
        if self.use_window:
            if hidden_states.shape[1] > 1: # prefilling stage
                num_windows = hidden_states.shape[1] // self.q_window_size
                
                key_states = self.k_proj_a(hidden_states)   # [bsz, q_len, hidden_size] -> [bsz, q_len, k_origin_dim + k_compressed_dim]
                key_states_origin = key_states[:, :, : self.num_key_value_heads * self.head_dim]  # [bsz, q_len, k_origin_dim]
                key_states_compressed = key_states[:, :, self.num_key_value_heads * self.head_dim: ]  # [bsz, q_len, k_compressed_dim]
                # full windows
                key_states_compressed_full = key_states_compressed[:, : num_windows * self.q_window_size, :]
                # residual
                key_states_compressed_res = key_states_compressed[:, num_windows * self.q_window_size:, :]
                key_states_origin_res = key_states_origin[:, num_windows * self.q_window_size:, :]
                del key_states  
                del key_states_compressed          
                
                value_states = self.v_proj_a(hidden_states) 
                value_states_origin = value_states[:, :, : self.num_key_value_heads * self.head_dim]
                value_states_compressed = value_states[:, :, self.num_key_value_heads * self.head_dim: ]
                # full windows   
                value_states_compressed_full = value_states_compressed[:, : num_windows * self.q_window_size, :]
                # residual
                value_states_compressed_res = value_states_compressed[:, num_windows * self.q_window_size:, :]
                value_states_origin_res = value_states_origin[:, num_windows * self.q_window_size:, :]
                del value_states     
                del value_states_compressed    
                
                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                
                key_states_origin = key_states_origin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [bsz, q_len, k_origin_dim] -> [bsz, num_heads, q_len, head_dim]
                key_states_compressed_full = key_states_compressed_full.view(bsz, num_windows * self.q_window_size, self.num_key_value_heads, self.k_compressed_dim_per_head).transpose(1, 2) 
                key_states_compressed_res = key_states_compressed_res.view(bsz, q_len - num_windows * self.q_window_size, self.num_key_value_heads, self.k_compressed_dim_per_head).transpose(1, 2)   
                key_states_origin_res = key_states_origin_res.view(bsz, q_len - num_windows * self.q_window_size, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
                
                value_states_origin = value_states_origin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  
                value_states_compressed_full = value_states_compressed_full.view(bsz, num_windows * self.q_window_size, self.num_key_value_heads, self.v_compressed_dim_per_head).transpose(1, 2)  
                value_states_compressed_res = value_states_compressed_res.view(bsz, q_len - num_windows * self.q_window_size, self.num_key_value_heads, self.v_compressed_dim_per_head).transpose(1, 2)
                value_states_origin_res = value_states_origin_res.view(bsz, q_len - num_windows * self.q_window_size, self.num_key_value_heads, self.head_dim).transpose(1, 2)  
    
                # quant compressed full window
                if self.k_bits < 16 and num_windows > 0:
                    key_states_compressed_full = self.quantizer_k.apply(key_states_compressed_full, self.k_bits, self.q_window_size)    # use per-channel quant for key cache
                if self.v_bits < 16 and num_windows > 0:
                    value_states_compressed_full = self.quantizer_v.apply(value_states_compressed_full, self.v_bits, self.v_compressed_dim_per_head)    # use per-token quant for value cache
                    
                # save kv cache (compressed+quant full window & compressed residual & origin residual)
                past_key_value = getattr(self, "past_key_value", past_key_value)
                if past_key_value is not None:
                    # sin and cos are specific to RoPE models; position_ids needed for the static cache
                    # no cache_kwargs is needed for DynamicCache, we just drop it
                    cache_kwargs = None # {"sin": sin, "cos": cos, "cache_position": cache_position}
                    _, _ = past_key_value.update_compressed_quant(key_states_compressed_full, value_states_compressed_full, self.layer_idx, cache_kwargs)    # [bsz, num_heads, k_len, v_compressed_dim_per_head] 
                    _, _ = past_key_value.update_compressed(key_states_compressed_res, value_states_compressed_res, self.layer_idx, cache_kwargs)
                    _, _ = past_key_value.update_origin(key_states_origin_res, value_states_origin_res, self.layer_idx, cache_kwargs)

                key_states_all, value_states_all = key_states_origin, value_states_origin   
                
            else:       # decoding stage
                past_key_value = getattr(self, "past_key_value", past_key_value)
                past_len = past_key_value.get_seq_length(layer_idx=self.layer_idx)   
                
                key_states = self.k_proj_a(hidden_states)   # [bsz, q_len, hidden_size] -> [bsz, q_len, k_origin_dim + k_compressed_dim]
                key_states_origin = key_states[:, :, : self.num_key_value_heads * self.head_dim]  # [bsz, q_len, k_origin_dim]
                key_states_compressed = key_states[:, :, self.num_key_value_heads * self.head_dim: ]  # [bsz, q_len, k_compressed_dim]
                del key_states  
                    
                value_states = self.v_proj_a(hidden_states) 
                value_states_origin = value_states[:, :, : self.num_key_value_heads * self.head_dim]
                value_states_compressed = value_states[:, :, self.num_key_value_heads * self.head_dim: ]   
                del value_states     
                
                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                
                key_states_origin = key_states_origin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [bsz, q_len, k_origin_dim] -> [bsz, num_heads, q_len, head_dim]
                key_states_compressed = key_states_compressed.view(bsz, q_len, self.num_key_value_heads, self.k_compressed_dim_per_head).transpose(1, 2)  
                
                value_states_origin = value_states_origin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  
                value_states_compressed = value_states_compressed.view(bsz, q_len, self.num_key_value_heads, self.v_compressed_dim_per_head).transpose(1, 2)  
                
                if past_key_value is not None: 
                    if (past_len + 1) % self.q_window_size == 0:    # after adding the current token, it just fills a window           
                        # sin and cos are specific to RoPE models; position_ids needed for the static cache
                        # no cache_kwargs is needed for DynamicCache, we just drop it
                        cache_kwargs = None # {"sin": sin, "cos": cos, "cache_position": cache_position}
                        # firstly store the
                        # store and get the current full window's original KV Cache
                        key_states_origin_window, value_states_origin_window = past_key_value.update_origin(key_states_origin, value_states_origin, self.layer_idx, cache_kwargs)
                        # store and get the current full window's compressed KV Cache
                        key_states_compressed_window, value_states_compressed_window = past_key_value.update_compressed(key_states_compressed, value_states_compressed, self.layer_idx, cache_kwargs)
                        # clear the original KV Cache in memory
                        past_key_value.clear_origin(self.layer_idx)
                        # quant the current full window
                        if self.k_bits < 16:
                            key_states_compressed_window = self.quantizer_k.apply(key_states_compressed_window, self.k_bits, self.q_window_size)    # use per-channel quant for key cache
                        if self.v_bits < 16:
                            value_states_compressed_window = self.quantizer_v.apply(value_states_compressed_window, self.v_bits, self.v_compressed_dim_per_head)    # use per-token quant for value cache 
                        # clear the compressed but un-quant KV Cache in memory
                        past_key_value.clear_compressed(self.layer_idx)
                        # store the quantized version of the current full window's KV Cache, and get all the full windows
                        key_states_compressed_all, value_states_compressed_all = past_key_value.update_compressed_quant(key_states_compressed_window, value_states_compressed_window, self.layer_idx, cache_kwargs)
                        
                        # reconstruct k-cache and v-cache 
                        k_len, v_len = key_states_compressed_all.shape[2], value_states_compressed_all.shape[2]
                        key_states_all = self.k_proj_b(key_states_compressed_all.transpose(1, 2).reshape(bsz, k_len, self.k_compressed_dim)).view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
                        # [bsz, num_heads, k_len, k_compressed_dim_per_head] -> [bsz, k_len, k_compressed_dim] -> [bsz, k_len, num_heads * head_dim] -> [bsz, num_heads, k_len, head_dim]
                        value_states_all = self.v_proj_b(value_states_compressed_all.transpose(1, 2).reshape(bsz, v_len, self.v_compressed_dim)).view(bsz, v_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
                        
                        # replace the last window's reconstructed KV Cache with the full-precision one
                        key_states_all[:, :, -self.q_window_size: ] = key_states_origin_window
                        value_states_all[:, :, -self.q_window_size: ] = value_states_origin_window
                        
                    else:   # adding the current token doesn't fill a window
                        # sin and cos are specific to RoPE models; position_ids needed for the static cache
                        # no cache_kwargs is needed for DynamicCache, we just drop it
                        cache_kwargs = None # {"sin": sin, "cos": cos, "cache_position": cache_position}
                        # firstly store the
                        # store the current token's original KV Cache, and get the current window's original KV Cache
                        key_states_origin_window, value_states_origin_window = past_key_value.update_origin(key_states_origin, value_states_origin, self.layer_idx, cache_kwargs)
                        # store the current token's compressed KV Cache (without quant)
                        _, _ = past_key_value.update_compressed(key_states_compressed, value_states_compressed, self.layer_idx, cache_kwargs)
                        
                        # get the compressed+quant full windows' KV Cache
                        key_states_compressed_all, value_states_compressed_all = past_key_value.update_compressed_quant(None, None, self.layer_idx, cache_kwargs)
                        if key_states_compressed_all.shape[2] == 0:     # there haven't been any full windows in KV Cache
                            key_states_all, value_states_all = key_states_origin_window, value_states_origin_window
                        else:
                            # reconstruct k-cache and v-cache 
                            k_len, v_len = key_states_compressed_all.shape[2], value_states_compressed_all.shape[2]
                            key_states_all = self.k_proj_b(key_states_compressed_all.transpose(1, 2).reshape(bsz, k_len, self.k_compressed_dim)).view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
                            # [bsz, num_heads, k_len, k_compressed_dim_per_head] -> [bsz, k_len, k_compressed_dim] -> [bsz, k_len, num_heads * head_dim] -> [bsz, num_heads, k_len, head_dim]
                            value_states_all = self.v_proj_b(value_states_compressed_all.transpose(1, 2).reshape(bsz, v_len, self.v_compressed_dim)).view(bsz, v_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
                                
                            # concat the last window's original kv with the reconstructed full windows
                            key_states_all = torch.cat((key_states_all, key_states_origin_window.to(key_states_all.device)), dim=2)
                            value_states_all = torch.cat((value_states_all, value_states_origin_window.to(value_states_all.device)), dim=2)
                            # # replace the last window's reconstructed KV Cache with the full-precision one
                            # cur_window_len = key_states_origin_window.shape[2]
                            # key_states_all[:, :, -cur_window_len: ] = key_states_origin_window
                            # value_states_all[:, :, -cur_window_len: ] = value_states_origin_window
            
        else:   # do not use window
            key_states = self.k_proj_a(hidden_states)   # [bsz, q_len, hidden_size] -> [bsz, q_len, k_origin_dim + k_compressed_dim]
            key_states_origin = key_states[:, :, : self.num_key_value_heads * self.head_dim]  # [bsz, q_len, k_origin_dim]
            key_states_compressed = key_states[:, :, self.num_key_value_heads * self.head_dim: ]  # [bsz, q_len, k_compressed_dim]
            del key_states

            value_states = self.v_proj_a(hidden_states) 
            value_states_origin = value_states[:, :, : self.num_key_value_heads * self.head_dim]
            value_states_compressed = value_states[:, :, self.num_key_value_heads * self.head_dim: ]      
            del value_states     

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            key_states_origin = key_states_origin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [bsz, q_len, k_origin_dim] -> [bsz, num_heads, q_len, head_dim]
            key_states_compressed = key_states_compressed.view(bsz, q_len, self.num_key_value_heads, self.k_compressed_dim_per_head).transpose(1, 2)  # [bsz, q_len, k_compressed_dim] -> [bsz, num_heads, q_len, k_compressed_dim_per_head]
            
            value_states_origin = value_states_origin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  
            value_states_compressed = value_states_compressed.view(bsz, q_len, self.num_key_value_heads, self.v_compressed_dim_per_head).transpose(1, 2)  

            # ================================ KV Cache quantization
            if self.k_bits < 16:
                key_states_compressed = self.quantizer_k.apply(key_states_compressed, self.k_bits, self.q_window_size if self.use_window else self.k_compressed_dim_per_head)
            if self.v_bits < 16:
                value_states_compressed = self.quantizer_v.apply(value_states_compressed, self.v_bits, self.v_compressed_dim_per_head)
            # =================================================================

            # ======================= store and get the compressed kvcache (without RoPE)
            past_key_value = getattr(self, "past_key_value", past_key_value)
            past_len = past_key_value.get_seq_length(layer_idx=self.layer_idx)
            if past_key_value is not None:
                # sin and cos are specific to RoPE models; position_ids needed for the static cache
                # no cache_kwargs is needed for DynamicCache, we just drop it
                cache_kwargs = None # {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states_compressed_all, value_states_compressed_all = past_key_value.update(key_states_compressed, value_states_compressed, self.layer_idx, cache_kwargs)    # [bsz, num_heads, k_len, v_compressed_dim_per_head]
            
            # =============================================================

            # =================== reconstruct k-cache and v-cache 
            k_len, v_len = key_states_compressed_all.shape[2], value_states_compressed_all.shape[2]
            key_states_all = self.k_proj_b(key_states_compressed_all.transpose(1, 2).reshape(bsz, k_len, self.k_compressed_dim)).view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
            # [bsz, num_heads, k_len, k_compressed_dim_per_head] -> [bsz, k_len, k_compressed_dim] -> [bsz, k_len, num_heads * head_dim] -> [bsz, num_heads, k_len, head_dim]
            value_states_all = self.v_proj_b(value_states_compressed_all.transpose(1, 2).reshape(bsz, v_len, self.v_compressed_dim)).view(bsz, v_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
            # ==================================================

            # ================================== replace the last token's reconstructed kvcache with the full-precision one
            if past_len == 0:    # prefill
                key_states_all, value_states_all = key_states_origin, value_states_origin
            else:   # decode, replace the last token's kvcache with the original key or value cache
                assert key_states_origin.shape[2] == 1
                key_states_all[:, :, -1] = key_states_origin[: , :, 0]
                value_states_all[:, :, -1] = value_states_origin[: , :, 0]



        # ===================== get the RoPE for the compressed kvcache 
        # note: because the stored kcache in past_key_value has not been positional embeded, 
        # so all the past kcache should be positional embeded here.
        if position_ids.shape[1] == 1:  # decode stage, original position_id would be like `[[275]]`, we want it to be like `[[0,1,...,275]]`
            position_ids_all = torch.tensor([list(range(position_ids[0][0]+1)) for _ in range(position_ids.shape[0])]).to(position_ids.dtype).to(position_ids.device)
        else:   # prefill stage
            position_ids_all = position_ids
            
        cos_q, sin_q = self.rotary_emb(value_states_all, position_ids)  # q is not affected
        cos_k, sin_k = self.rotary_emb(value_states_all, position_ids_all)
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos_q, sin_q)
        key_states_all, _ = apply_rotary_pos_emb(key_states_all, key_states_all, cos_k, sin_k)
        # ============================================================================

        key_states_all = repeat_kv(key_states_all, self.num_key_value_groups)
        value_states_all = repeat_kv(value_states_all, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states_all.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            if cache_position is not None:
                causal_mask = attention_mask[:, :, cache_position, : key_states_all.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states_all)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class LlamaFlashAttention2ForCSKV(LlamaAttentionForCSKV):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        
        if self.use_window:
            if hidden_states.shape[1] > 1: # prefilling stage
                num_windows = hidden_states.shape[1] // self.q_window_size
                
                key_states = self.k_proj_a(hidden_states)   # [bsz, q_len, hidden_size] -> [bsz, q_len, k_origin_dim + k_compressed_dim]
                key_states_origin = key_states[:, :, : self.num_key_value_heads * self.head_dim]  # [bsz, q_len, k_origin_dim]
                key_states_compressed = key_states[:, :, self.num_key_value_heads * self.head_dim: ]  # [bsz, q_len, k_compressed_dim]
                # full windows
                key_states_compressed_full = key_states_compressed[:, : num_windows * self.q_window_size, :]
                # residual
                key_states_compressed_res = key_states_compressed[:, num_windows * self.q_window_size:, :]
                key_states_origin_res = key_states_origin[:, num_windows * self.q_window_size:, :]
                del key_states  
                del key_states_compressed          
                
                value_states = self.v_proj_a(hidden_states) 
                value_states_origin = value_states[:, :, : self.num_key_value_heads * self.head_dim]
                value_states_compressed = value_states[:, :, self.num_key_value_heads * self.head_dim: ]
                # full windows   
                value_states_compressed_full = value_states_compressed[:, : num_windows * self.q_window_size, :]
                # residual
                value_states_compressed_res = value_states_compressed[:, num_windows * self.q_window_size:, :]
                value_states_origin_res = value_states_origin[:, num_windows * self.q_window_size:, :]
                del value_states     
                del value_states_compressed    
                
                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                
                key_states_origin = key_states_origin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [bsz, q_len, k_origin_dim] -> [bsz, num_heads, q_len, head_dim]
                key_states_compressed_full = key_states_compressed_full.view(bsz, num_windows * self.q_window_size, self.num_key_value_heads, self.k_compressed_dim_per_head).transpose(1, 2) 
                key_states_compressed_res = key_states_compressed_res.view(bsz, q_len - num_windows * self.q_window_size, self.num_key_value_heads, self.k_compressed_dim_per_head).transpose(1, 2)   
                key_states_origin_res = key_states_origin_res.view(bsz, q_len - num_windows * self.q_window_size, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
                
                value_states_origin = value_states_origin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  
                value_states_compressed_full = value_states_compressed_full.view(bsz, num_windows * self.q_window_size, self.num_key_value_heads, self.v_compressed_dim_per_head).transpose(1, 2)  
                value_states_compressed_res = value_states_compressed_res.view(bsz, q_len - num_windows * self.q_window_size, self.num_key_value_heads, self.v_compressed_dim_per_head).transpose(1, 2)
                value_states_origin_res = value_states_origin_res.view(bsz, q_len - num_windows * self.q_window_size, self.num_key_value_heads, self.head_dim).transpose(1, 2)  
    
                # quant compressed full window
                if self.k_bits < 16 and num_windows > 0:
                    key_states_compressed_full = self.quantizer_k.apply(key_states_compressed_full, self.k_bits, self.q_window_size)    # use per-channel quant for key cache
                if self.v_bits < 16 and num_windows > 0:
                    value_states_compressed_full = self.quantizer_v.apply(value_states_compressed_full, self.v_bits, self.v_compressed_dim_per_head)    # use per-token quant for value cache
                    
                # save kv cache (compressed+quant full window & compressed residual & origin residual)
                past_key_value = getattr(self, "past_key_value", past_key_value)
                if past_key_value is not None:
                    # sin and cos are specific to RoPE models; position_ids needed for the static cache
                    # no cache_kwargs is needed for DynamicCache, we just drop it
                    cache_kwargs = None # {"sin": sin, "cos": cos, "cache_position": cache_position}
                    _, _ = past_key_value.update_compressed_quant(key_states_compressed_full, value_states_compressed_full, self.layer_idx, cache_kwargs)    # [bsz, num_heads, k_len, v_compressed_dim_per_head] 
                    _, _ = past_key_value.update_compressed(key_states_compressed_res, value_states_compressed_res, self.layer_idx, cache_kwargs)
                    _, _ = past_key_value.update_origin(key_states_origin_res, value_states_origin_res, self.layer_idx, cache_kwargs)

                key_states_all, value_states_all = key_states_origin, value_states_origin   
                
            else:       # decoding stage
                past_key_value = getattr(self, "past_key_value", past_key_value)
                past_len = past_key_value.get_seq_length(layer_idx=self.layer_idx)   
                
                key_states = self.k_proj_a(hidden_states)   # [bsz, q_len, hidden_size] -> [bsz, q_len, k_origin_dim + k_compressed_dim]
                key_states_origin = key_states[:, :, : self.num_key_value_heads * self.head_dim]  # [bsz, q_len, k_origin_dim]
                key_states_compressed = key_states[:, :, self.num_key_value_heads * self.head_dim: ]  # [bsz, q_len, k_compressed_dim]
                del key_states  
                    
                value_states = self.v_proj_a(hidden_states) 
                value_states_origin = value_states[:, :, : self.num_key_value_heads * self.head_dim]
                value_states_compressed = value_states[:, :, self.num_key_value_heads * self.head_dim: ]   
                del value_states     
                
                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                
                key_states_origin = key_states_origin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [bsz, q_len, k_origin_dim] -> [bsz, num_heads, q_len, head_dim]
                key_states_compressed = key_states_compressed.view(bsz, q_len, self.num_key_value_heads, self.k_compressed_dim_per_head).transpose(1, 2)  
                
                value_states_origin = value_states_origin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  
                value_states_compressed = value_states_compressed.view(bsz, q_len, self.num_key_value_heads, self.v_compressed_dim_per_head).transpose(1, 2)  
                
                if past_key_value is not None: 
                    if (past_len + 1) % self.q_window_size == 0:    # after adding the current token, it just fills a window           
                        # sin and cos are specific to RoPE models; position_ids needed for the static cache
                        # no cache_kwargs is needed for DynamicCache, we just drop it
                        cache_kwargs = None # {"sin": sin, "cos": cos, "cache_position": cache_position}
                        # firstly store the
                        # store and get the current full window's original KV Cache
                        key_states_origin_window, value_states_origin_window = past_key_value.update_origin(key_states_origin, value_states_origin, self.layer_idx, cache_kwargs)
                        # store and get the current full window's compressed KV Cache
                        key_states_compressed_window, value_states_compressed_window = past_key_value.update_compressed(key_states_compressed, value_states_compressed, self.layer_idx, cache_kwargs)
                        # clear the original KV Cache in memory
                        past_key_value.clear_origin(self.layer_idx)
                        # quant the current full window
                        if self.k_bits < 16:
                            key_states_compressed_window = self.quantizer_k.apply(key_states_compressed_window, self.k_bits, self.q_window_size)    # use per-channel quant for key cache
                        if self.v_bits < 16:
                            value_states_compressed_window = self.quantizer_v.apply(value_states_compressed_window, self.v_bits, self.v_compressed_dim_per_head)    # use per-token quant for value cache 
                        # clear the compressed but un-quant KV Cache in memory
                        past_key_value.clear_compressed(self.layer_idx)
                        # store the quantized version of the current full window's KV Cache, and get all the full windows
                        key_states_compressed_all, value_states_compressed_all = past_key_value.update_compressed_quant(key_states_compressed_window, value_states_compressed_window, self.layer_idx, cache_kwargs)
                        
                        # reconstruct k-cache and v-cache 
                        k_len, v_len = key_states_compressed_all.shape[2], value_states_compressed_all.shape[2]
                        key_states_all = self.k_proj_b(key_states_compressed_all.transpose(1, 2).reshape(bsz, k_len, self.k_compressed_dim)).view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
                        # [bsz, num_heads, k_len, k_compressed_dim_per_head] -> [bsz, k_len, k_compressed_dim] -> [bsz, k_len, num_heads * head_dim] -> [bsz, num_heads, k_len, head_dim]
                        value_states_all = self.v_proj_b(value_states_compressed_all.transpose(1, 2).reshape(bsz, v_len, self.v_compressed_dim)).view(bsz, v_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
                        
                        # replace the last window's reconstructed KV Cache with the full-precision one
                        key_states_all[:, :, -self.q_window_size: ] = key_states_origin_window
                        value_states_all[:, :, -self.q_window_size: ] = value_states_origin_window
                        
                    else:   # adding the current token doesn't fill a window
                        # sin and cos are specific to RoPE models; position_ids needed for the static cache
                        # no cache_kwargs is needed for DynamicCache, we just drop it
                        cache_kwargs = None # {"sin": sin, "cos": cos, "cache_position": cache_position}
                        # firstly store the
                        # store the current token's original KV Cache, and get the current window's original KV Cache
                        key_states_origin_window, value_states_origin_window = past_key_value.update_origin(key_states_origin, value_states_origin, self.layer_idx, cache_kwargs)
                        # store the current token's compressed KV Cache (without quant)
                        _, _ = past_key_value.update_compressed(key_states_compressed, value_states_compressed, self.layer_idx, cache_kwargs)
                        
                        # get the compressed+quant full windows' KV Cache
                        key_states_compressed_all, value_states_compressed_all = past_key_value.update_compressed_quant(None, None, self.layer_idx, cache_kwargs)
                        if key_states_compressed_all.shape[2] == 0:     # there haven't been any full windows in KV Cache
                            key_states_all, value_states_all = key_states_origin_window, value_states_origin_window
                        else:
                            # reconstruct k-cache and v-cache 
                            k_len, v_len = key_states_compressed_all.shape[2], value_states_compressed_all.shape[2]
                            key_states_all = self.k_proj_b(key_states_compressed_all.transpose(1, 2).reshape(bsz, k_len, self.k_compressed_dim)).view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
                            # [bsz, num_heads, k_len, k_compressed_dim_per_head] -> [bsz, k_len, k_compressed_dim] -> [bsz, k_len, num_heads * head_dim] -> [bsz, num_heads, k_len, head_dim]
                            value_states_all = self.v_proj_b(value_states_compressed_all.transpose(1, 2).reshape(bsz, v_len, self.v_compressed_dim)).view(bsz, v_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
                                
                            # concat the last window's original kv with the reconstructed full windows
                            key_states_all = torch.cat((key_states_all, key_states_origin_window.to(key_states_all.device)), dim=2)
                            value_states_all = torch.cat((value_states_all, value_states_origin_window.to(value_states_all.device)), dim=2)
                            # # replace the last window's reconstructed KV Cache with the full-precision one
                            # cur_window_len = key_states_origin_window.shape[2]
                            # key_states_all[:, :, -cur_window_len: ] = key_states_origin_window
                            # value_states_all[:, :, -cur_window_len: ] = value_states_origin_window
            
        else:   # do not use window
            key_states = self.k_proj_a(hidden_states)   # [bsz, q_len, hidden_size] -> [bsz, q_len, k_origin_dim + k_compressed_dim]
            key_states_origin = key_states[:, :, : self.num_key_value_heads * self.head_dim]  # [bsz, q_len, k_origin_dim]
            key_states_compressed = key_states[:, :, self.num_key_value_heads * self.head_dim: ]  # [bsz, q_len, k_compressed_dim]
            del key_states

            value_states = self.v_proj_a(hidden_states) 
            value_states_origin = value_states[:, :, : self.num_key_value_heads * self.head_dim]
            value_states_compressed = value_states[:, :, self.num_key_value_heads * self.head_dim: ]      
            del value_states     

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            key_states_origin = key_states_origin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [bsz, q_len, k_origin_dim] -> [bsz, num_heads, q_len, head_dim]
            key_states_compressed = key_states_compressed.view(bsz, q_len, self.num_key_value_heads, self.k_compressed_dim_per_head).transpose(1, 2)  # [bsz, q_len, k_compressed_dim] -> [bsz, num_heads, q_len, k_compressed_dim_per_head]
            
            value_states_origin = value_states_origin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  
            value_states_compressed = value_states_compressed.view(bsz, q_len, self.num_key_value_heads, self.v_compressed_dim_per_head).transpose(1, 2)  

            # ================================ KV Cache quantization
            if self.k_bits < 16:
                key_states_compressed = self.quantizer_k.apply(key_states_compressed, self.k_bits, self.q_window_size if self.use_window else self.k_compressed_dim_per_head)
            if self.v_bits < 16:
                value_states_compressed = self.quantizer_v.apply(value_states_compressed, self.v_bits, self.v_compressed_dim_per_head)
            # =================================================================

            # ======================= store and get the compressed kvcache (without RoPE)
            past_key_value = getattr(self, "past_key_value", past_key_value)
            past_len = past_key_value.get_seq_length(layer_idx=self.layer_idx)
            if past_key_value is not None:
                # sin and cos are specific to RoPE models; position_ids needed for the static cache
                # no cache_kwargs is needed for DynamicCache, we just drop it
                cache_kwargs = None # {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states_compressed_all, value_states_compressed_all = past_key_value.update(key_states_compressed, value_states_compressed, self.layer_idx, cache_kwargs)    # [bsz, num_heads, k_len, v_compressed_dim_per_head]
            
            # =============================================================

            # =================== reconstruct k-cache and v-cache 
            k_len, v_len = key_states_compressed_all.shape[2], value_states_compressed_all.shape[2]
            key_states_all = self.k_proj_b(key_states_compressed_all.transpose(1, 2).reshape(bsz, k_len, self.k_compressed_dim)).view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
            # [bsz, num_heads, k_len, k_compressed_dim_per_head] -> [bsz, k_len, k_compressed_dim] -> [bsz, k_len, num_heads * head_dim] -> [bsz, num_heads, k_len, head_dim]
            value_states_all = self.v_proj_b(value_states_compressed_all.transpose(1, 2).reshape(bsz, v_len, self.v_compressed_dim)).view(bsz, v_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
            # ==================================================

            # ================================== replace the last token's reconstructed kvcache with the full-precision one
            if past_len == 0:    # prefill
                key_states_all, value_states_all = key_states_origin, value_states_origin
            else:   # decode, replace the last token's kvcache with the original key or value cache
                assert key_states_origin.shape[2] == 1
                key_states_all[:, :, -1] = key_states_origin[: , :, 0]
                value_states_all[:, :, -1] = value_states_origin[: , :, 0]



        # ===================== get the RoPE for the compressed kvcache 
        # note: because the stored kcache in past_key_value has not been positional embeded, 
        # so all the past kcache should be positional embeded here.
        if position_ids.shape[1] == 1:  # decode stage, original position_id would be like `[[275]]`, we want it to be like `[[0,1,...,275]]`
            position_ids_all = torch.tensor([list(range(position_ids[0][0]+1)) for _ in range(position_ids.shape[0])]).to(position_ids.dtype).to(position_ids.device)
        else:   # prefill stage
            position_ids_all = position_ids
            
        cos_q, sin_q = self.rotary_emb(value_states_all, position_ids)  # q is not affected
        cos_k, sin_k = self.rotary_emb(value_states_all, position_ids_all)
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos_q, sin_q)
        key_states_all, _ = apply_rotary_pos_emb(key_states_all, key_states_all, cos_k, sin_k)
        # ============================================================================

        key_states_all = repeat_kv(key_states_all, self.num_key_value_groups)
        value_states_all = repeat_kv(value_states_all, self.num_key_value_groups)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states_all = key_states_all.transpose(1, 2)
        value_states_all = value_states_all.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logging.warn(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states_all = key_states_all.to(target_dtype)
            value_states_all = value_states_all.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states_all, value_states_all, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

class LlamaSdpaAttentionForCSKV(LlamaAttentionForCSKV):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logging.warn(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
        
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        
        if self.use_window:
            if hidden_states.shape[1] > 1: # prefilling stage
                num_windows = hidden_states.shape[1] // self.q_window_size
                
                key_states = self.k_proj_a(hidden_states)   # [bsz, q_len, hidden_size] -> [bsz, q_len, k_origin_dim + k_compressed_dim]
                key_states_origin = key_states[:, :, : self.num_key_value_heads * self.head_dim]  # [bsz, q_len, k_origin_dim]
                key_states_compressed = key_states[:, :, self.num_key_value_heads * self.head_dim: ]  # [bsz, q_len, k_compressed_dim]
                # full windows
                key_states_compressed_full = key_states_compressed[:, : num_windows * self.q_window_size, :]
                # residual
                key_states_compressed_res = key_states_compressed[:, num_windows * self.q_window_size:, :]
                key_states_origin_res = key_states_origin[:, num_windows * self.q_window_size:, :]
                del key_states  
                del key_states_compressed          
                
                value_states = self.v_proj_a(hidden_states) 
                value_states_origin = value_states[:, :, : self.num_key_value_heads * self.head_dim]
                value_states_compressed = value_states[:, :, self.num_key_value_heads * self.head_dim: ]
                # full windows   
                value_states_compressed_full = value_states_compressed[:, : num_windows * self.q_window_size, :]
                # residual
                value_states_compressed_res = value_states_compressed[:, num_windows * self.q_window_size:, :]
                value_states_origin_res = value_states_origin[:, num_windows * self.q_window_size:, :]
                del value_states     
                del value_states_compressed    
                
                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                
                key_states_origin = key_states_origin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [bsz, q_len, k_origin_dim] -> [bsz, num_heads, q_len, head_dim]
                key_states_compressed_full = key_states_compressed_full.view(bsz, num_windows * self.q_window_size, self.num_key_value_heads, self.k_compressed_dim_per_head).transpose(1, 2) 
                key_states_compressed_res = key_states_compressed_res.view(bsz, q_len - num_windows * self.q_window_size, self.num_key_value_heads, self.k_compressed_dim_per_head).transpose(1, 2)   
                key_states_origin_res = key_states_origin_res.view(bsz, q_len - num_windows * self.q_window_size, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
                
                value_states_origin = value_states_origin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  
                value_states_compressed_full = value_states_compressed_full.view(bsz, num_windows * self.q_window_size, self.num_key_value_heads, self.v_compressed_dim_per_head).transpose(1, 2)  
                value_states_compressed_res = value_states_compressed_res.view(bsz, q_len - num_windows * self.q_window_size, self.num_key_value_heads, self.v_compressed_dim_per_head).transpose(1, 2)
                value_states_origin_res = value_states_origin_res.view(bsz, q_len - num_windows * self.q_window_size, self.num_key_value_heads, self.head_dim).transpose(1, 2)  
    
                # quant compressed full window
                if self.k_bits < 16 and num_windows > 0:
                    key_states_compressed_full = self.quantizer_k.apply(key_states_compressed_full, self.k_bits, self.q_window_size)    # use per-channel quant for key cache
                if self.v_bits < 16 and num_windows > 0:
                    value_states_compressed_full = self.quantizer_v.apply(value_states_compressed_full, self.v_bits, self.v_compressed_dim_per_head)    # use per-token quant for value cache
                    
                # save kv cache (compressed+quant full window & compressed residual & origin residual)
                past_key_value = getattr(self, "past_key_value", past_key_value)
                if past_key_value is not None:
                    # sin and cos are specific to RoPE models; position_ids needed for the static cache
                    # no cache_kwargs is needed for DynamicCache, we just drop it
                    cache_kwargs = None # {"sin": sin, "cos": cos, "cache_position": cache_position}
                    _, _ = past_key_value.update_compressed_quant(key_states_compressed_full, value_states_compressed_full, self.layer_idx, cache_kwargs)    # [bsz, num_heads, k_len, v_compressed_dim_per_head] 
                    _, _ = past_key_value.update_compressed(key_states_compressed_res, value_states_compressed_res, self.layer_idx, cache_kwargs)
                    _, _ = past_key_value.update_origin(key_states_origin_res, value_states_origin_res, self.layer_idx, cache_kwargs)

                key_states_all, value_states_all = key_states_origin, value_states_origin   
                
            else:       # decoding stage
                past_key_value = getattr(self, "past_key_value", past_key_value)
                past_len = past_key_value.get_seq_length(layer_idx=self.layer_idx)   
                
                key_states = self.k_proj_a(hidden_states)   # [bsz, q_len, hidden_size] -> [bsz, q_len, k_origin_dim + k_compressed_dim]
                key_states_origin = key_states[:, :, : self.num_key_value_heads * self.head_dim]  # [bsz, q_len, k_origin_dim]
                key_states_compressed = key_states[:, :, self.num_key_value_heads * self.head_dim: ]  # [bsz, q_len, k_compressed_dim]
                del key_states  
                    
                value_states = self.v_proj_a(hidden_states) 
                value_states_origin = value_states[:, :, : self.num_key_value_heads * self.head_dim]
                value_states_compressed = value_states[:, :, self.num_key_value_heads * self.head_dim: ]   
                del value_states     
                
                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                
                key_states_origin = key_states_origin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [bsz, q_len, k_origin_dim] -> [bsz, num_heads, q_len, head_dim]
                key_states_compressed = key_states_compressed.view(bsz, q_len, self.num_key_value_heads, self.k_compressed_dim_per_head).transpose(1, 2)  
                
                value_states_origin = value_states_origin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  
                value_states_compressed = value_states_compressed.view(bsz, q_len, self.num_key_value_heads, self.v_compressed_dim_per_head).transpose(1, 2)  
                
                if past_key_value is not None: 
                    if (past_len + 1) % self.q_window_size == 0:    # after adding the current token, it just fills a window           
                        # sin and cos are specific to RoPE models; position_ids needed for the static cache
                        # no cache_kwargs is needed for DynamicCache, we just drop it
                        cache_kwargs = None # {"sin": sin, "cos": cos, "cache_position": cache_position}
                        # firstly store the
                        # store and get the current full window's original KV Cache
                        key_states_origin_window, value_states_origin_window = past_key_value.update_origin(key_states_origin, value_states_origin, self.layer_idx, cache_kwargs)
                        # store and get the current full window's compressed KV Cache
                        key_states_compressed_window, value_states_compressed_window = past_key_value.update_compressed(key_states_compressed, value_states_compressed, self.layer_idx, cache_kwargs)
                        # clear the original KV Cache in memory
                        past_key_value.clear_origin(self.layer_idx)
                        # quant the current full window
                        if self.k_bits < 16:
                            key_states_compressed_window = self.quantizer_k.apply(key_states_compressed_window, self.k_bits, self.q_window_size)    # use per-channel quant for key cache
                        if self.v_bits < 16:
                            value_states_compressed_window = self.quantizer_v.apply(value_states_compressed_window, self.v_bits, self.v_compressed_dim_per_head)    # use per-token quant for value cache 
                        # clear the compressed but un-quant KV Cache in memory
                        past_key_value.clear_compressed(self.layer_idx)
                        # store the quantized version of the current full window's KV Cache, and get all the full windows
                        key_states_compressed_all, value_states_compressed_all = past_key_value.update_compressed_quant(key_states_compressed_window, value_states_compressed_window, self.layer_idx, cache_kwargs)
                        
                        # reconstruct k-cache and v-cache 
                        k_len, v_len = key_states_compressed_all.shape[2], value_states_compressed_all.shape[2]
                        key_states_all = self.k_proj_b(key_states_compressed_all.transpose(1, 2).reshape(bsz, k_len, self.k_compressed_dim)).view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
                        # [bsz, num_heads, k_len, k_compressed_dim_per_head] -> [bsz, k_len, k_compressed_dim] -> [bsz, k_len, num_heads * head_dim] -> [bsz, num_heads, k_len, head_dim]
                        value_states_all = self.v_proj_b(value_states_compressed_all.transpose(1, 2).reshape(bsz, v_len, self.v_compressed_dim)).view(bsz, v_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
                        
                        # replace the last window's reconstructed KV Cache with the full-precision one
                        key_states_all[:, :, -self.q_window_size: ] = key_states_origin_window
                        value_states_all[:, :, -self.q_window_size: ] = value_states_origin_window
                        
                    else:   # adding the current token doesn't fill a window
                        # sin and cos are specific to RoPE models; position_ids needed for the static cache
                        # no cache_kwargs is needed for DynamicCache, we just drop it
                        cache_kwargs = None # {"sin": sin, "cos": cos, "cache_position": cache_position}
                        # firstly store the
                        # store the current token's original KV Cache, and get the current window's original KV Cache
                        key_states_origin_window, value_states_origin_window = past_key_value.update_origin(key_states_origin, value_states_origin, self.layer_idx, cache_kwargs)
                        # store the current token's compressed KV Cache (without quant)
                        _, _ = past_key_value.update_compressed(key_states_compressed, value_states_compressed, self.layer_idx, cache_kwargs)
                        
                        # get the compressed+quant full windows' KV Cache
                        key_states_compressed_all, value_states_compressed_all = past_key_value.update_compressed_quant(None, None, self.layer_idx, cache_kwargs)
                        if key_states_compressed_all.shape[2] == 0:     # there haven't been any full windows in KV Cache
                            key_states_all, value_states_all = key_states_origin_window, value_states_origin_window
                        else:
                            # reconstruct k-cache and v-cache 
                            k_len, v_len = key_states_compressed_all.shape[2], value_states_compressed_all.shape[2]
                            key_states_all = self.k_proj_b(key_states_compressed_all.transpose(1, 2).reshape(bsz, k_len, self.k_compressed_dim)).view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
                            # [bsz, num_heads, k_len, k_compressed_dim_per_head] -> [bsz, k_len, k_compressed_dim] -> [bsz, k_len, num_heads * head_dim] -> [bsz, num_heads, k_len, head_dim]
                            value_states_all = self.v_proj_b(value_states_compressed_all.transpose(1, 2).reshape(bsz, v_len, self.v_compressed_dim)).view(bsz, v_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
                                
                            # concat the last window's original kv with the reconstructed full windows
                            key_states_all = torch.cat((key_states_all, key_states_origin_window.to(key_states_all.device)), dim=2)
                            value_states_all = torch.cat((value_states_all, value_states_origin_window.to(value_states_all.device)), dim=2)
                            # # replace the last window's reconstructed KV Cache with the full-precision one
                            # cur_window_len = key_states_origin_window.shape[2]
                            # key_states_all[:, :, -cur_window_len: ] = key_states_origin_window
                            # value_states_all[:, :, -cur_window_len: ] = value_states_origin_window
            
        else:   # do not use window
            key_states = self.k_proj_a(hidden_states)   # [bsz, q_len, hidden_size] -> [bsz, q_len, k_origin_dim + k_compressed_dim]
            key_states_origin = key_states[:, :, : self.num_key_value_heads * self.head_dim]  # [bsz, q_len, k_origin_dim]
            key_states_compressed = key_states[:, :, self.num_key_value_heads * self.head_dim: ]  # [bsz, q_len, k_compressed_dim]
            del key_states

            value_states = self.v_proj_a(hidden_states) 
            value_states_origin = value_states[:, :, : self.num_key_value_heads * self.head_dim]
            value_states_compressed = value_states[:, :, self.num_key_value_heads * self.head_dim: ]      
            del value_states     

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            key_states_origin = key_states_origin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [bsz, q_len, k_origin_dim] -> [bsz, num_heads, q_len, head_dim]
            key_states_compressed = key_states_compressed.view(bsz, q_len, self.num_key_value_heads, self.k_compressed_dim_per_head).transpose(1, 2)  # [bsz, q_len, k_compressed_dim] -> [bsz, num_heads, q_len, k_compressed_dim_per_head]
            
            value_states_origin = value_states_origin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  
            value_states_compressed = value_states_compressed.view(bsz, q_len, self.num_key_value_heads, self.v_compressed_dim_per_head).transpose(1, 2)  

            # ================================ KV Cache quantization
            if self.k_bits < 16:
                key_states_compressed = self.quantizer_k.apply(key_states_compressed, self.k_bits, self.q_window_size if self.use_window else self.k_compressed_dim_per_head)
            if self.v_bits < 16:
                value_states_compressed = self.quantizer_v.apply(value_states_compressed, self.v_bits, self.v_compressed_dim_per_head)
            # =================================================================

            # ======================= store and get the compressed kvcache (without RoPE)
            past_key_value = getattr(self, "past_key_value", past_key_value)
            past_len = past_key_value.get_seq_length(layer_idx=self.layer_idx)
            if past_key_value is not None:
                # sin and cos are specific to RoPE models; position_ids needed for the static cache
                # no cache_kwargs is needed for DynamicCache, we just drop it
                cache_kwargs = None # {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states_compressed_all, value_states_compressed_all = past_key_value.update(key_states_compressed, value_states_compressed, self.layer_idx, cache_kwargs)    # [bsz, num_heads, k_len, v_compressed_dim_per_head]
            
            # =============================================================

            # =================== reconstruct k-cache and v-cache 
            k_len, v_len = key_states_compressed_all.shape[2], value_states_compressed_all.shape[2]
            key_states_all = self.k_proj_b(key_states_compressed_all.transpose(1, 2).reshape(bsz, k_len, self.k_compressed_dim)).view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
            # [bsz, num_heads, k_len, k_compressed_dim_per_head] -> [bsz, k_len, k_compressed_dim] -> [bsz, k_len, num_heads * head_dim] -> [bsz, num_heads, k_len, head_dim]
            value_states_all = self.v_proj_b(value_states_compressed_all.transpose(1, 2).reshape(bsz, v_len, self.v_compressed_dim)).view(bsz, v_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
            # ==================================================

            # ================================== replace the last token's reconstructed kvcache with the full-precision one
            if past_len == 0:    # prefill
                key_states_all, value_states_all = key_states_origin, value_states_origin
            else:   # decode, replace the last token's kvcache with the original key or value cache
                assert key_states_origin.shape[2] == 1
                key_states_all[:, :, -1] = key_states_origin[: , :, 0]
                value_states_all[:, :, -1] = value_states_origin[: , :, 0]



        # ===================== get the RoPE for the compressed kvcache 
        # note: because the stored kcache in past_key_value has not been positional embeded, 
        # so all the past kcache should be positional embeded here.
        if position_ids.shape[1] == 1:  # decode stage, original position_id would be like `[[275]]`, we want it to be like `[[0,1,...,275]]`
            position_ids_all = torch.tensor([list(range(position_ids[0][0]+1)) for _ in range(position_ids.shape[0])]).to(position_ids.dtype).to(position_ids.device)
        else:   # prefill stage
            position_ids_all = position_ids
            
        cos_q, sin_q = self.rotary_emb(value_states_all, position_ids)  # q is not affected
        cos_k, sin_k = self.rotary_emb(value_states_all, position_ids_all)
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos_q, sin_q)
        key_states_all, _ = apply_rotary_pos_emb(key_states_all, key_states_all, cos_k, sin_k)
        # ============================================================================

        key_states_all = repeat_kv(key_states_all, self.num_key_value_groups)
        value_states_all = repeat_kv(value_states_all, self.num_key_value_groups)

        causal_mask = attention_mask
        causal_mask = attention_mask
        if attention_mask is not None and cache_position is not None:
            causal_mask = causal_mask[:, :, cache_position, : key_states_all.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states_all = key_states_all.contiguous()
            value_states_all = value_states_all.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states_all,
            value_states_all,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


# <<<<<<<<<<<<<<<<<<<<<< Channel Reduction for SVD + Parallel >>>>>>>>>>>>>>>>>>>>>>>




# >>>>>>>>>>>>>>>>>>>>>> Channel Reduction for SVD + Parallel Train >>>>>>>>>>>>>>>>>>>>>>>>
class LlamaAttentionForCSKVTrain(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None, args=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            print(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

        self.k_compressed_dim_per_head = args.k_compressed_dim
        self.v_compressed_dim_per_head = args.v_compressed_dim
        self.k_compressed_dim = self.k_compressed_dim_per_head * self.num_key_value_heads # Note: here k_compressed_dim should be num_heads*head_dim, eg. 32*64
        self.v_compressed_dim = self.v_compressed_dim_per_head * self.num_key_value_heads

        assert not config.attention_bias, "Bias is not supported in SVD initialization"
        self.k_proj_a = nn.Linear(self.hidden_size, self.k_compressed_dim, bias=False)    # U_k\cdot\sqrt{\Sigma_k}
        self.k_proj_b = nn.Linear(self.k_compressed_dim, self.num_key_value_heads * self.head_dim, bias=False)    # \sqrt{\Sigma_k}\cdot V_k^T
        self.v_proj_a = nn.Linear(self.hidden_size, self.v_compressed_dim, bias=False)
        self.v_proj_b = nn.Linear(self.v_compressed_dim, self.num_key_value_heads * self.head_dim, bias=False)

        self.mse_criterion = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        self.k_mse_loss = 0.0
        self.v_mse_loss = 0.0

        # quantization
        # window based
        self.use_window = args.use_window    
        self.q_window_size = args.q_window_size
        if self.use_window:  # according to KIVI, we only use per-channel quant for Key Cache, while keeping per-token quant for Value Cache
            self.quantizer_k = KVQuantizerChannel
        else:
            self.quantizer_k = KVQuantizer
        self.quantizer_v = KVQuantizer
        
        self.k_bits = args.k_bits
        self.v_bits = args.v_bits 

        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            raise NotImplementedError("`pretraining_tp` > 1 has not been `supported yet")
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            if self.use_window: # use window
                assert hidden_states.shape[1] % self.q_window_size == 0, f"Training sequence length (got {hidden_states.shape[1]}) must be multiple of q_window_size ({self.q_window_size})"

            query_states = self.q_proj(hidden_states)
            
            key_states_origin = self.k_proj(hidden_states) # [bsz, q_len, hidden_size] -> [bsz, q_len, k_origin_dim]
            key_states_compressed = self.k_proj_a(hidden_states)  # # [bsz, q_len, hidden_size] -> [bsz, q_len, k_compressed_dim]
            
            value_states_origin = self.v_proj(hidden_states)
            value_states_compressed = self.v_proj_a(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            key_states_origin = key_states_origin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [bsz, q_len, k_origin_dim] -> [bsz, num_heads, q_len, head_dim]
            key_states_compressed = key_states_compressed.view(bsz, q_len, self.num_key_value_heads, self.k_compressed_dim_per_head).transpose(1, 2)  # [bsz, q_len, k_compressed_dim] -> [bsz, num_heads, q_len, k_compressed_dim_per_head]
            
            value_states_origin = value_states_origin.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  
            value_states_compressed = value_states_compressed.view(bsz, q_len, self.num_key_value_heads, self.v_compressed_dim_per_head).transpose(1, 2)  

            # ================================ KV Cache quantization
            if self.k_bits < 16:
                key_states_compressed = self.quantizer_k.apply(key_states_compressed, self.k_bits, self.q_window_size if self.use_window else self.k_compressed_dim_per_head)
            if self.v_bits < 16:
                value_states_compressed = self.quantizer_v.apply(value_states_compressed, self.v_bits, self.v_compressed_dim_per_head)
            # =================================================================

            # ======================= store and get the compressed kvcache (without RoPE)
            # NO USE in training
            # past_key_value = getattr(self, "past_key_value", past_key_value)
            # past_len = past_key_value.get_seq_length(layer_idx=self.layer_idx)
            # if past_key_value is not None:
            #     # sin and cos are specific to RoPE models; position_ids needed for the static cache
            #     # no cache_kwargs is needed for DynamicCache, we just drop it
            #     cache_kwargs = None # {"sin": sin, "cos": cos, "cache_position": cache_position}
            #     key_states_compressed_all, value_states_compressed_all = past_key_value.update(key_states_compressed, value_states_compressed, self.layer_idx, cache_kwargs)    # [bsz, num_heads, k_len, v_compressed_dim_per_head]
            
            # =============================================================

            # =================== reconstruct k-cache and v-cache 
            k_len, v_len = key_states_compressed.shape[2], value_states_compressed.shape[2]
            key_states_reconstructed= self.k_proj_b(key_states_compressed.transpose(1, 2).reshape(bsz, k_len, self.k_compressed_dim)).view(bsz, k_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
            # [bsz, num_heads, k_len, k_compressed_dim_per_head] -> [bsz, k_len, k_compressed_dim] -> [bsz, k_len, num_heads * head_dim] -> [bsz, num_heads, k_len, head_dim]
            value_states_reconstructed = self.v_proj_b(value_states_compressed.transpose(1, 2).reshape(bsz, v_len, self.v_compressed_dim)).view(bsz, v_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
            # ==================================================

        # ================================== replace the last token's reconstructed kvcache with the full-precision one
        # NO Use in training
        # if past_len == 0:    # prefill
        #     key_states_all, value_states_all = key_states_origin, value_states_origin
        # else:   # decode, replace the last token's kvcache with the original key or value cache
        #     assert key_states_origin.shape[2] == 1
        #     key_states_all[:, :, -1] = key_states_origin[: , :, 0]
        #     value_states_all[:, :, -1] = value_states_origin[: , :, 0]


        # ===================== get the RoPE for the compressed kvcache 
        # note: because the stored kcache in past_key_value has not been positional embeded, 
        # so all the past kcache should be positional embeded here.
        if position_ids.shape[1] == 1:  # decode stage, original position_id would be like `[[275]]`, we want it to be like `[[0,1,...,275]]`
            position_ids_all = torch.tensor([list(range(position_ids[0][0]+1)) for _ in range(position_ids.shape[0])]).to(position_ids.dtype).to(position_ids.device)
        else:   # prefill stage
            position_ids_all = position_ids

        cos_q, sin_q = self.rotary_emb(value_states_origin, position_ids)  # q is not affected
        cos_k, sin_k = self.rotary_emb(value_states_origin, position_ids_all)
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos_q, sin_q)
        key_states_origin, _ = apply_rotary_pos_emb(key_states_origin, key_states_origin, cos_k, sin_k)
        key_states_reconstructed, _ = apply_rotary_pos_emb(key_states_reconstructed, key_states_reconstructed, cos_k, sin_k)
        # ============================================================================

        # ================= compute mse loss
        self.k_mse_loss = self.mse_criterion(key_states_origin, key_states_reconstructed)
        self.v_mse_loss = self.mse_criterion(value_states_origin, value_states_reconstructed)
        # ====================================

        key_states_origin = repeat_kv(key_states_origin, self.num_key_value_groups)
        value_states_origin = repeat_kv(value_states_origin, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states_origin.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            if cache_position is not None:
                causal_mask = attention_mask[:, :, cache_position, : key_states_origin.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states_origin)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
# <<<<<<<<<<<<<<<<<<<<<< Channel Reduction for SVD + Parallel Train <<<<<<<<<<<<<<<<<<<<<<<<







