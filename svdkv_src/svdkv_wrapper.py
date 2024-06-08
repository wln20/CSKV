from tqdm import tqdm
import os
import logging
import torch
from functools import partial
from types import MethodType
from collections import OrderedDict
from transformers.cache_utils import DynamicCache
from svdkv_src.modeling.modeling_llama import (
    LlamaAttentionForSVDKV,
    LlamaAttentionForSVDKVTrain,
    forward_llama,
    forward_llama_model,
    prepare_inputs_for_generation_llama,
)

from svdkv_src.utils.training_utils import (
    init_svd_parallel_inference, 
    init_svd_parallel_inference_load_ckpt,
    init_svd,
    init_asvd,
    init_asvd_parallel_inference
)

from svdkv_src.utils.general_utils import get_name

def get_svdkv_model(model, args):
    """
    for inference, would load trained checkpoints for inferencing
    """

    logging.info(f"* Arguments: {args}")
    logging.info(f"Getting svdkv model for {args.model_id} ...")

    # ckpt path
    args.k_compressor_ckpt = os.path.join(args.kv_ckpt_root_path, args.model_id, f"{get_name(args)}_k.ckpt")
    args.v_compressor_ckpt = os.path.join(args.kv_ckpt_root_path, args.model_id, f"{get_name(args)}_v.ckpt")

    if args.use_asvd:
        with open(os.path.join(args.asvd_calib_root, args.model_id, 'calib_input_distribution_abs_mean.pt'), 'rb') as f:
            asvd_calib_data_all = torch.load(f)

    # replace the original attention modules
    if 'llama' in model.config.architectures[0].lower() or 'vicuna' in model.config.architectures[0].lower() or 'longchat' in model.config.architectures[0].lower() or 'baichuan' in model.config.architectures[0].lower():
        # replace the original model with the custom one
        model.model.use_window = args.use_window
        model.model.q_window_size = args.q_window_size
        model.use_window = args.use_window
        model.q_window_size = args.q_window_size
        model.is_training = False
        model.forward = MethodType(forward_llama, model)
        model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation_llama, model)
        model.model.forward = MethodType(forward_llama_model, model.model)
        
        # replace attention modules
        for i in tqdm(range(len(model.model.layers))):
            block = model.model.layers[i]
            new_attn = LlamaAttentionForSVDKV(model.config, block.self_attn.layer_idx, args).to(block.self_attn.q_proj.weight.dtype).to(block.self_attn.q_proj.weight.device)
            new_attn_state_dict = block.self_attn.state_dict().copy()  # odict_keys(['q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight'])
            if args.use_asvd:
                if args.use_init_params:
                    new_attn_state_dict['k_proj_a.weight'], new_attn_state_dict['k_proj_b.weight'] = init_asvd_parallel_inference(new_attn_state_dict['k_proj.weight'], calib_data=asvd_calib_data_all[i][0], compressed_dim_per_head=args.k_compressed_dim, num_heads=model.config.num_key_value_heads)
                    new_attn_state_dict['v_proj_a.weight'], new_attn_state_dict['v_proj_b.weight'] = init_asvd_parallel_inference(new_attn_state_dict['v_proj.weight'], calib_data=asvd_calib_data_all[i][1], compressed_dim_per_head=args.v_compressed_dim, num_heads=model.config.num_key_value_heads)
                else:
                    with open(args.k_compressor_ckpt, 'rb') as fk, open(args.v_compressor_ckpt, 'rb') as fv:
                        k_proj_a_state_dict_part, k_proj_b_state_dict_part = torch.load(fk)[block.self_attn.layer_idx]   # k_proj_a_state_dict.keys() = odict_keys(['weight']), k_proj_b_state_dict.keys() = odict_keys(['weight'])
                        v_proj_a_state_dict_part, v_proj_b_state_dict_part = torch.load(fv)[block.self_attn.layer_idx]   
                        new_attn_state_dict['k_proj_a.weight'], new_attn_state_dict['k_proj_b.weight'] = init_svd_parallel_inference_load_ckpt(new_attn_state_dict['k_proj.weight'], k_proj_a_state_dict_part['weight'], k_proj_b_state_dict_part['weight'])
                        new_attn_state_dict['v_proj_a.weight'], new_attn_state_dict['v_proj_b.weight'] = init_svd_parallel_inference_load_ckpt(new_attn_state_dict['v_proj.weight'], v_proj_a_state_dict_part['weight'], v_proj_b_state_dict_part['weight'])           
            
            else:
                if args.use_init_params:
                    new_attn_state_dict['k_proj_a.weight'], new_attn_state_dict['k_proj_b.weight'] = init_svd_parallel_inference(new_attn_state_dict['k_proj.weight'], compressed_dim_per_head=args.k_compressed_dim, num_heads=model.config.num_key_value_heads)
                    new_attn_state_dict['v_proj_a.weight'], new_attn_state_dict['v_proj_b.weight'] = init_svd_parallel_inference(new_attn_state_dict['v_proj.weight'], compressed_dim_per_head=args.v_compressed_dim, num_heads=model.config.num_key_value_heads)
                else:
                    with open(args.k_compressor_ckpt, 'rb') as fk, open(args.v_compressor_ckpt, 'rb') as fv:
                        k_proj_a_state_dict_part, k_proj_b_state_dict_part = torch.load(fk)[block.self_attn.layer_idx]   # k_proj_a_state_dict.keys() = odict_keys(['weight']), k_proj_b_state_dict.keys() = odict_keys(['weight'])
                        v_proj_a_state_dict_part, v_proj_b_state_dict_part = torch.load(fv)[block.self_attn.layer_idx]   
                        new_attn_state_dict['k_proj_a.weight'], new_attn_state_dict['k_proj_b.weight'] = init_svd_parallel_inference_load_ckpt(new_attn_state_dict['k_proj.weight'], k_proj_a_state_dict_part['weight'], k_proj_b_state_dict_part['weight'])
                        new_attn_state_dict['v_proj_a.weight'], new_attn_state_dict['v_proj_b.weight'] = init_svd_parallel_inference_load_ckpt(new_attn_state_dict['v_proj.weight'], v_proj_a_state_dict_part['weight'], v_proj_b_state_dict_part['weight'])                
            del(new_attn_state_dict['k_proj.weight'])
            del(new_attn_state_dict['v_proj.weight'])
                    
            new_attn.load_state_dict(new_attn_state_dict)
            block.self_attn = new_attn
    else:
        raise NotImplementedError(f"Not support model architecture: {model.config.architectures[0]}")

    torch.cuda.empty_cache()
    return model

def get_svdkv_model_train(model, args):

    logging.debug(f"* Arguments: {args}")
    logging.info(f"Getting svdkv model (train) for {args.model_id} ...")

    if args.use_asvd:
        with open(os.path.join(args.asvd_calib_root, args.model_id, 'calib_input_distribution_abs_mean.pt'), 'rb') as f:
            asvd_calib_data_all = torch.load(f)

    # replace the original attention modules
    if 'llama' in model.config.architectures[0].lower() or 'vicuna' in model.config.architectures[0].lower() or 'longchat' in model.config.architectures[0].lower() or 'baichuan' in model.config.architectures[0].lower():
        # replace the original model with the custom one
        model.model.use_window = args.use_window
        model.model.q_window_size = args.q_window_size
        model.use_window = args.use_window
        model.q_window_size = args.q_window_size
        model.is_training = True
        model.forward = MethodType(forward_llama, model)
        model.model.forward = MethodType(forward_llama_model, model.model)
        
        # replace attention modules
        for i in tqdm(range(len(model.model.layers))):
            block = model.model.layers[i]
               
            new_attn = LlamaAttentionForSVDKVTrain(model.config, block.self_attn.layer_idx, args).to(block.self_attn.q_proj.weight.dtype).to(block.self_attn.q_proj.weight.device)
            new_attn_state_dict = block.self_attn.state_dict().copy()  # odict_keys(['q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight'])
            if args.use_asvd:
                new_attn_state_dict['k_proj_a.weight'], new_attn_state_dict['k_proj_b.weight'] = init_asvd(new_attn_state_dict['k_proj.weight'], asvd_calib_data_all[i][0], compressed_dim_per_head=args.k_compressed_dim, num_heads=model.config.num_key_value_heads)
                new_attn_state_dict['v_proj_a.weight'], new_attn_state_dict['v_proj_b.weight'] = init_asvd(new_attn_state_dict['v_proj.weight'], asvd_calib_data_all[i][1], compressed_dim_per_head=args.v_compressed_dim, num_heads=model.config.num_key_value_heads)            
            else:
                new_attn_state_dict['k_proj_a.weight'], new_attn_state_dict['k_proj_b.weight'] = init_svd(new_attn_state_dict['k_proj.weight'], compressed_dim_per_head=args.k_compressed_dim, num_heads=model.config.num_key_value_heads)
                new_attn_state_dict['v_proj_a.weight'], new_attn_state_dict['v_proj_b.weight'] = init_svd(new_attn_state_dict['v_proj.weight'], compressed_dim_per_head=args.v_compressed_dim, num_heads=model.config.num_key_value_heads)
          
            new_attn.load_state_dict(new_attn_state_dict)
            block.self_attn = new_attn

        # set gradient requirements
        num_trainable_params = 0
        num_all_params = 0
        for name, param in model.named_parameters():
            if 'k_proj_a' in name or 'v_proj_a' in name or 'k_proj_b' in name or 'v_proj_b' in name:
                param.requires_grad = True
                num_trainable_params += param.numel()    
            else:
                param.requires_grad = False  
            num_all_params += param.numel()              

        assert num_trainable_params > 0, "No parameter is trainable, exit training process."
        logging.info(f"* Number of trainable arguments: {num_trainable_params / 1024**2:.2f} M")
        logging.info(f"* Proportion of trainable arguments: {100 * num_trainable_params / num_all_params:.4f} %")
        torch.cuda.empty_cache()

    else:
        raise NotImplementedError(f"Not support model architecture: {model.config.architectures[0]}")

    return model

