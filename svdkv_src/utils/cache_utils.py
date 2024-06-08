from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers.cache_utils import Cache


class DynamicCacheWithWindow(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self, q_window_size=32) -> None:
        self.key_cache_compressed_quant: List[torch.Tensor] = []
        self.value_cache_compressed_quant: List[torch.Tensor] = []
        self.key_cache_compressed: List[torch.Tensor] = []
        self.value_cache_compressed: List[torch.Tensor] = []
        self.key_cache_origin: List[torch.Tensor] = []
        self.value_cache_origin: List[torch.Tensor] = []
        self.seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        
        self.q_window_size = q_window_size

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache_compressed_quant[layer_idx], self.value_cache_compressed_quant[layer_idx], self.key_cache_compressed[layer_idx], self.value_cache_compressed[layer_idx], self.key_cache_origin[layer_idx], self.value_cache_origin[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache_compressed_quant[layer_idx], self.value_cache_compressed_quant[layer_idx], self.key_cache_compressed[layer_idx], self.value_cache_compressed[layer_idx], self.key_cache_origin[layer_idx], self.value_cache_origin[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        assert len(self.key_cache_compressed_quant) == len(self.key_cache_compressed) and len(self.key_cache_compressed_quant) == len(self.key_cache_origin)
        return len(self.key_cache_compressed_quant)


    def update_compressed_quant(
        self,
        key_states_compressed_quant: torch.Tensor,
        value_states_compressed_quant: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states_compressed_quant is None:
            return self.key_cache_compressed_quant[layer_idx], self.value_cache_compressed_quant[layer_idx]
        
        assert key_states_compressed_quant.shape[-2] % self.q_window_size == 0
        # Update the number of seen tokens
        if layer_idx == 0:
            self.seen_tokens += key_states_compressed_quant.shape[-2]
        
        # Update the cache
        if len(self.key_cache_compressed_quant) <= layer_idx:
            self.key_cache_compressed_quant.append(key_states_compressed_quant)
            self.value_cache_compressed_quant.append(value_states_compressed_quant)
        else:
            self.key_cache_compressed_quant[layer_idx] = torch.cat([self.key_cache_compressed_quant[layer_idx], key_states_compressed_quant.to(self.key_cache_compressed_quant[layer_idx].device)], dim=-2)
            self.value_cache_compressed_quant[layer_idx] = torch.cat([self.value_cache_compressed_quant[layer_idx], value_states_compressed_quant.to(self.value_cache_compressed_quant[layer_idx].device)], dim=-2)

        return self.key_cache_compressed_quant[layer_idx], self.value_cache_compressed_quant[layer_idx]

    def update_compressed(
        self,
        key_states_compressed: torch.Tensor,
        value_states_compressed: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Do not update the number of seen tokens here, because states_compressed is always parallel with states_origin
        # Update the cache
        if len(self.key_cache_compressed) <= layer_idx:
            self.key_cache_compressed.append(key_states_compressed)
            self.value_cache_compressed.append(value_states_compressed)
        else:
            self.key_cache_compressed[layer_idx] = torch.cat([self.key_cache_compressed[layer_idx], key_states_compressed.to(self.key_cache_compressed[layer_idx].device)], dim=-2)
            self.value_cache_compressed[layer_idx] = torch.cat([self.value_cache_compressed[layer_idx], value_states_compressed.to(self.value_cache_compressed[layer_idx].device)], dim=-2)

        return self.key_cache_compressed[layer_idx], self.value_cache_compressed[layer_idx]

    def clear_compressed(self, layer_idx):
        key_shape = self.key_cache_compressed[layer_idx].shape
        value_shape = self.value_cache_compressed[layer_idx].shape
        key_dtype = self.key_cache_origin[layer_idx].dtype
        value_dtype = self.value_cache_origin[layer_idx].dtype
        # replace with an empty placeholder with the original shape
        self.key_cache_compressed[layer_idx] = torch.empty(key_shape[0], key_shape[1], 0, key_shape[3], dtype=key_dtype).to(self.key_cache_compressed[layer_idx].device)
        self.value_cache_compressed[layer_idx] = torch.empty(value_shape[0], value_shape[1], 0, value_shape[3], dtype=value_dtype).to(self.value_cache_compressed[layer_idx].device)

    def update_origin(
        self,
        key_states_origin: torch.Tensor,
        value_states_origin: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self.seen_tokens += key_states_origin.shape[-2]
        
        # Update the cache
        if len(self.key_cache_origin) <= layer_idx:
            self.key_cache_origin.append(key_states_origin)
            self.value_cache_origin.append(value_states_origin)
        else:
            self.key_cache_origin[layer_idx] = torch.cat([self.key_cache_origin[layer_idx], key_states_origin.to(self.key_cache_origin[layer_idx].device)], dim=-2)
            self.value_cache_origin[layer_idx] = torch.cat([self.value_cache_origin[layer_idx], value_states_origin.to(self.value_cache_origin[layer_idx].device)], dim=-2)

        return self.key_cache_origin[layer_idx], self.value_cache_origin[layer_idx]
    
    def clear_origin(self, layer_idx):
        assert self.key_cache_origin[layer_idx].shape[2] == self.q_window_size
        if layer_idx == 0:
            self.seen_tokens -= self.q_window_size
        key_shape = self.key_cache_origin[layer_idx].shape
        value_shape = self.value_cache_origin[layer_idx].shape
        key_dtype = self.key_cache_origin[layer_idx].dtype
        value_dtype = self.value_cache_origin[layer_idx].dtype
        # replace with an empty placeholder with the original shape
        self.key_cache_origin[layer_idx] = torch.empty(key_shape[0], key_shape[1], 0, key_shape[3], dtype=key_dtype).to(self.key_cache_origin[layer_idx].device)
        self.value_cache_origin[layer_idx] = torch.empty(value_shape[0], value_shape[1], 0, value_shape[3], dtype=value_dtype).to(self.value_cache_origin[layer_idx].device)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache_compressed_quant) <= layer_idx:
            return 0
        return self.key_cache_compressed_quant[layer_idx].shape[-2] + self.key_cache_origin[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        raise NotImplementedError("Beam search has not been supported yet")
        # for layer_idx in range(len(self.key_cache)):
        #     device = self.key_cache[layer_idx].device
        #     self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
        #     device = self.value_cache[layer_idx].device
        #     self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache_compressed_quant[layer_idx], self.value_cache_compressed_quant[layer_idx], self.key_cache_compressed[layer_idx], self.value_cache_compressed[layer_idx], self.key_cache_origin[layer_idx], self.value_cache_origin[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, q_window_size=32) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls(q_window_size)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states_compressed_quant, value_states_compressed_quant, key_states_compressed, value_states_compressed, key_states_origin, value_states_origin = past_key_values[layer_idx]
                cache.update_compressed_quant(key_states_compressed_quant, value_states_compressed_quant, layer_idx)
                cache.update_compressed(key_states_compressed, value_states_compressed, layer_idx)
                cache.update_origin(key_states_origin, value_states_origin, layer_idx)
        return cache