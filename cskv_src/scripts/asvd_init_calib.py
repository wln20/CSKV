# Mean
# alpha = 0.5

import argparse
import os
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from datasets import load_dataset

def get_calib_data(name, tokenizer, model_id, nsamples, seqlen=2048, seed=3):
    if name == "c4":
        traindata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        tot_text = "\n\n".join(traindata["text"])
    elif name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tot_text = "\n\n".join(traindata["text"])
    elif name == 'pile_subset':
        data = load_dataset(
            "ola13/small-the_pile",
            split='train',
        )
        tot_text = '\n\n'.join(data['text'])   
    else:
        raise NotImplementedError
    print(f"tot_text={len(tot_text)}")
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset

@torch.no_grad()
def calib_input_distribution(model, model_id, cache_root_path, calib_loader, method='abs_mean', use_cache=True):
    cache_file = os.path.join(cache_root_path, model_id, f"calib_input_distribution_{method}.pt")

    os.makedirs(os.path.split(cache_file)[0])
    if os.path.exists(cache_file) and use_cache:
        all_scaling_diag_matrix = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            # if isinstance(module, nn.Linear):
            if "k_proj" in name or "v_proj" in name:
                module.scaling_diag_matrix = all_scaling_diag_matrix[name].to(
                    module.weight.device
                )
        return
    model.eval()
    # set hook for k_proj and v_prok layers

    def hook(module, input, output):
        if "abs_mean" in method:
            abs_mean = input[0].abs().mean(dim=-2).detach().view(-1)
            module.scaling_diag_matrix += abs_mean
        elif "abs_max" in method:
            abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
            module.scaling_diag_matrix = torch.where(
                abs_max > module.scaling_diag_matrix,
                abs_max,
                module.scaling_diag_matrix,
            )
        # abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
        # module.scaling_diag_matrix += abs_max

    for name, module in model.named_modules():
        if "k_proj" in name or "v_proj" in name:
            module.scaling_diag_matrix = 0
            module.register_forward_hook(hook)

    # get activation distribution
    for batch in tqdm(calib_loader):
        # print(batch)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        model(**batch)

    # remove and save scaling_diag_matrix
    all_scaling_diag_matrix = {}
    for name, module in model.named_modules():
        if "k_proj" in name or "v_proj" in name:
            module._forward_hooks.clear()
            all_scaling_diag_matrix[name] = module.scaling_diag_matrix
    
    all_list = []
    for i in range(model.config.num_hidden_layers):
        all_list.append((all_scaling_diag_matrix[f'model.layers.{i}.self_attn.k_proj'], all_scaling_diag_matrix[f'model.layers.{i}.self_attn.v_proj']))
    torch.save(all_list, cache_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/share/datasets/public_models/lmsys_longchat-7b-v1.5-32k', help='path to the hf model')
    parser.add_argument('--model_id', default='longchat-7b-v1.5-32k', help='the name you give to the model')
    parser.add_argument('--max_seq_len', type=int, default=4096)
    parser.add_argument('--cache_root_path', default='../data/asvd_data/asvd_init_ckpts')

    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_path)
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}   # must set dtype to torch.float32 to avoid overflow

    model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    calib_loader = get_calib_data(name='pile_subset', tokenizer=tokenizer, model_id=args.model_id, nsamples=256, seqlen=args.max_seq_len) 

    calib_input_distribution(model=model, model_id=args.model_id, cache_root_path=args.cache_root_path, calib_loader=calib_loader, method="abs_mean",)