import argparse
import logging
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TextStreamer

# self import
from svdkv_src.svdkv_wrapper import (
    get_svdkv_model
)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='/share/datasets/public_models/lmsys_longchat-7b-v1.5-32k', help='path to the hf model')
parser.add_argument('--model_id', default='longchat-7b-v1.5-32k', help='the name you give to the model')
# parser.add_argument('--model_path', default='/share/datasets/public_models/Llama-2-7b-chat-hf', help='path to the hf model')
# parser.add_argument('--model_id', default='llama2-7b-chat', help='the name you give to the model')

parser.add_argument('--logging_level', type=str, default='DEBUG', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])

# attention implementation
parser.add_argument("--attn_impl", default='eager', choices=['eager', 'flash_attention_2', 'sdpa'])

# compression ratio
parser.add_argument('--k_compressed_dim', type=int, default=32)
parser.add_argument('--v_compressed_dim', type=int, default=96)

# baseline, use original model
parser.add_argument('--use_origin_model', action='store_true')

# baseline, pure svd result without fine-tuning
parser.add_argument('--use_init_params', action='store_true')

# ckpt after fine-tuning
parser.add_argument('--kv_ckpt_root_path', default='./svdkv_src/data/kvcache_compressor_checkpoints')

# svd config
parser.add_argument('--use_asvd', action='store_true')
parser.add_argument('--asvd_calib_root', default='./svdkv_src/data/asvd_data/asvd_init_ckpts')

# quant config
parser.add_argument('--k_bits', type=int, default=16)
parser.add_argument('--v_bits', type=int, default=16)

# window_based quantization
parser.add_argument('--use_window', action='store_true')
parser.add_argument('--q_window_size', default=32)

args = parser.parse_args()

# logging basic config
logging.basicConfig(format='[%(levelname)s]: %(message)s', level=eval(f'logging.{args.logging_level}'))

if __name__ == "__main__":

    # load model
    logging.info(f"* Loading model and tokenizer from {args.model_path} ...")
    # set config
    config = AutoConfig.from_pretrained(args.model_path)
    config.max_position_embeddings = 32768
    config._attn_implementation = args.attn_impl  
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}

    model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # get svdkv model
    if not args.use_origin_model:
        logging.info(f"* K Compressed dim: {args.k_compressed_dim}")
        logging.info(f"* V Compressed dim: {args.v_compressed_dim}")
        if args.use_init_params:
            logging.warning("* Using initial parameters of kv compressor")
        model = get_svdkv_model(model, args)
        
    else:
        logging.warning("* Using original model as a baseline")

    # simple evaluation
    logging.info("Generatng ...")
    print('='*40)

    with open("./svdkv_src/data/long_data/data_1.json", 'r') as f:
        import json
        long_text = json.load(f)
        long_text = "Passage: " + long_text['text'] + "\n\n" + "Question: " + long_text['question'] + "\n\n" + "Answer: "
            
    prompts = [
        long_text,
       # "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Some people got on a bus at the terminal. At the first bus stop, half of the people got down and 4 more people got in. Then at the second bus stop, 6 people got down and 8 more got in. If there were a total of 25 people heading to the third stop, how many people got on the bus at the terminal? ASSISTANT: " # 38
    ]

    # print(f'Prompts: {prompts}')
    # print('-'*40)
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    inputs['input_ids'] = inputs['input_ids'].to(model.device)
    inputs['attention_mask'] = inputs['attention_mask'].to(model.device)
    streamer = TextStreamer(tokenizer)
    output = model.generate(**inputs, use_cache=True, streamer=streamer, do_sample=False, top_p=None, temperature=None, max_new_tokens=400)#, top_p=0.95, top_k=60)

    # for i, prompt in enumerate(prompts):
    #     print(f"Output_{i}: {tokenizer.decode(output[i][inputs['input_ids'].shape[1]:])}")
    #     print('-'*20)
    # print('='*40)
