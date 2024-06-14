import argparse
import logging
import datetime
import os
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['WANDB_MODE'] = 'disabled'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from svdkv_src.svdkv_wrapper import get_svdkv_model_train
from svdkv_src.utils.training_utils import TrainerForSVDKV, save_kvcompressor_ckpt

from svdkv_src.utils.data_utils import get_dataset_for_trainer
from svdkv_src.utils.general_utils import get_name

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='/share/datasets/public_models/lmsys_longchat-7b-v1.5-32k', help='path to the hf model')
parser.add_argument('--model_id', default='longchat-7b-v1.5-32k', help='the name you give to the model')
parser.add_argument('--max_seq_len', type=int, default=4992)
parser.add_argument('--logging_level', type=str, default='DEBUG', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])

# compression ratio
parser.add_argument('--k_density', type=float, default=0.25, help=r'how much key cache remains after compression, e.g. 0.25 means 25% remains')
parser.add_argument('--v_density', type=float, default=0.75, help=r'how much value cache remains after compression, e.g. 0.25 means 25% remains')

# training
parser.add_argument('--training_dataset', default='pile_subset', choices=['wikitext2', 'pile_subset'])
parser.add_argument('--epoches', type=float, default=1)
parser.add_argument('--warmup_steps', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--ckpt_save_path', default='../data/kvcache_compressor_checkpoints')

# asvd
parser.add_argument('--use_asvd', action='store_true')
parser.add_argument('--asvd_calib_root', default='../data/asvd_data/asvd_init_ckpts')

# qat with svd_init_parallel_inference
parser.add_argument('--k_bits', type=int, default=16)
parser.add_argument('--v_bits', type=int, default=16)

# window_based quantization
parser.add_argument('--use_window', action='store_true')
parser.add_argument('--q_window_size', type=int, default=32)

args = parser.parse_args()

# basic config
logging.basicConfig(format='[%(levelname)s]: %(message)s', level=eval(f'logging.{args.logging_level}'))

current_datetime = datetime.datetime.now()
timestamp = current_datetime.strftime("%Y%m%d_%H%M%S")
run_name = f"{args.model_id}_{get_name(args)}_{timestamp}"

if __name__ == '__main__':
    
    # load model
    logging.info(f"* Loading model and tokenizer from {args.model_path} ...")
    # set config
    config = AutoConfig.from_pretrained(args.model_path)
    kwargs = {"torch_dtype": torch.float32, "device_map": "auto"}   # must set dtype to torch.float32 to avoid overflow

    model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # load dataset
    logging.info(f"* Loading dataset ...")   
    # preprocess dataset
    dataset = get_dataset_for_trainer(args.training_dataset, tokenizer, split='train', seqlen=args.max_seq_len)

    # get svdkv channel reduction model
    model = get_svdkv_model_train(model, args)

    # set training arguments
    training_args = TrainingArguments(
        output_dir='./tmp_trainer',
        overwrite_output_dir=True,
        run_name=run_name,
        save_strategy='no',
        per_device_train_batch_size=1,
        num_train_epochs=args.epoches,
        bf16=True,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        logging_strategy='steps',
        logging_steps=50,
    )

    logging.info("Start training ...")
    # set up trainer
    trainer = TrainerForSVDKV(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )


    # begin training
    trainer.train()

    # save results
    logging.info("* Saving results ...")
    for kv in ['k', 'v']:
        ckpt_save_path = os.path.join(args.ckpt_save_path, args.model_id, f"{get_name(args)}_{kv}.ckpt")
        os.makedirs(os.path.split(ckpt_save_path)[0], exist_ok=True)
        save_kvcompressor_ckpt(model, ckpt_save_path, kv, args=args)
