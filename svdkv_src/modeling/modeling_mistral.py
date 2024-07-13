import inspect
import logging
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    replace_return_docstrings,
)
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding, apply_rotary_pos_emb, repeat_kv

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)



_CONFIG_FOR_DOC = "MistralConfig"

from svdkv_src.utils.quant_utils import KVQuantizer, KVQuantizerChannel
from svdkv_src.utils.cache_utils import DynamicCacheWithWindow

def prepare_inputs_for_generation_mistral(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    # Omit tokens covered by past_key_values
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
        }
    )
    return model_inputs



def forward_mistral(
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
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, MistralForCausalLM

    >>> model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

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
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Ensure tensors are on the same device
        shift_labels = shift_labels.to(shift_logits.device)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits, shift_labels)

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
    
    
def forward_mistral_model(
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
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logging.warn(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0
    if use_cache:  # kept for BC (cache positions)
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if not isinstance(past_key_values, StaticCache):
            if self.use_window:
                past_key_values = DynamicCacheWithWindow.from_legacy_cache(past_key_values, self.q_window_size)
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_seq_length()

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
        is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        if is_padding_right:
            raise ValueError(
                "You are attempting to perform batched generation with padding_side='right'"
                " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
            )

    if self._attn_implementation == "flash_attention_2":
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    elif self._attn_implementation == "sdpa" and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

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
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
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
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


class MistralAttentionForSVDKV(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None, args=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logging.warn(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
            
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.k_compressed_dim_per_head = args.k_compressed_dim
        self.v_compressed_dim_per_head = args.v_compressed_dim
        self.k_compressed_dim = self.k_compressed_dim_per_head * self.num_key_value_heads # Note: here k_compressed_dim should be num_heads*head_dim, eg. 32*64
        self.v_compressed_dim = self.v_compressed_dim_per_head * self.num_key_value_heads

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

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
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
            
        cos_q, sin_q = self.rotary_emb(value_states_all, len(position_ids_all[0]))  # q is not affected
        cos_k, sin_k = self.rotary_emb(value_states_all, len(position_ids_all[0]))
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos_q, sin_q, position_ids)
        key_states_all, _ = apply_rotary_pos_emb(key_states_all, key_states_all, cos_k, sin_k, position_ids_all)
        # ============================================================================

        key_states_all = repeat_kv(key_states_all, self.num_key_value_groups)
        value_states_all = repeat_kv(value_states_all, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states_all.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # if attention_mask.size() != (bsz, 1, q_len, past_len):
            #     raise ValueError(
            #         f"Attention mask should be of size {(bsz, 1, q_len, past_len)}, but is {attention_mask.size()}"
            #     )

            attn_weights = attn_weights + attention_mask

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
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
    
class MistralSdpaAttentionForSVDKV(MistralAttentionForSVDKV):
    """
    Mistral attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `MistralAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from MistralAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logging.warn(
                "MistralModel is using MistralSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
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
            
        cos_q, sin_q = self.rotary_emb(value_states_all, len(position_ids_all[0]))  # q is not affected
        cos_k, sin_k = self.rotary_emb(value_states_all, len(position_ids_all[0]))
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos_q, sin_q, position_ids)
        key_states_all, _ = apply_rotary_pos_emb(key_states_all, key_states_all, cos_k, sin_k, position_ids_all)
        # ============================================================================

        key_states_all = repeat_kv(key_states_all, self.num_key_value_groups)
        value_states_all = repeat_kv(value_states_all, self.num_key_value_groups)

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states_all.contiguous()
            value_states = value_states_all.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states_all,
            value_states_all,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
    

# >>>>>>>>>>>>>>>>>> Train >>>>>>>>>>>>>>>>>>>>>>>>>>
class MistralAttentionForSVDKVTrain(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None, args=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logging.warn(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
            
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.k_compressed_dim_per_head = args.k_compressed_dim
        self.v_compressed_dim_per_head = args.v_compressed_dim
        self.k_compressed_dim = self.k_compressed_dim_per_head * self.num_key_value_heads # Note: here k_compressed_dim should be num_heads*head_dim, eg. 32*64
        self.v_compressed_dim = self.v_compressed_dim_per_head * self.num_key_value_heads

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

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()  
        if self.use_window:
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
        # NO Use in training
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


        # # ================================== replace the last token's reconstructed kvcache with the full-precision one
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
            
        cos_q, sin_q = self.rotary_emb(value_states_origin, len(position_ids_all[0]))  # q is not affected
        cos_k, sin_k = self.rotary_emb(value_states_origin, len(position_ids_all[0]))
           
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos_q, sin_q, position_ids)
        key_states_origin, _ = apply_rotary_pos_emb(key_states_origin, key_states_origin, cos_k, sin_k, position_ids_all)
        key_states_reconstructed, _ = apply_rotary_pos_emb(key_states_reconstructed, key_states_reconstructed, cos_k, sin_k, position_ids_all)
        # ============================================================================

        # ================= compute mse loss
        self.k_mse_loss = self.mse_criterion(key_states_origin, key_states_reconstructed)
        self.v_mse_loss = self.mse_criterion(value_states_origin, value_states_reconstructed)
        # ====================================

        key_states_origin = repeat_kv(key_states_origin, self.num_key_value_groups)
        value_states_origin = repeat_kv(value_states_origin, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states_origin.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # if attention_mask.size() != (bsz, 1, q_len, past_len):
            #     raise ValueError(
            #         f"Attention mask should be of size {(bsz, 1, q_len, past_len)}, but is {attention_mask.size()}"
            #     )

            attn_weights = attn_weights + attention_mask

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
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value