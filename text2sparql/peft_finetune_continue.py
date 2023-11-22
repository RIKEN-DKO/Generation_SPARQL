"""
This version continue training from a previous peft model
"""
import argparse
from datasets import Dataset, DatasetDict
from transformers import LlamaTokenizer, AutoModelForCausalLM
import bitsandbytes as bnb
import torch
import os
import json
import transformers
from peft import LoraConfig
# from incorrect_promt import SPARQLQueryGenerator
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM

def load_data_from_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return {key: [d[key] for d in data] for key in data[0].keys()}


def generate_prompt(example):
    question = example['question']
    sparql = example["sparql"]
    prompt = f"### INSTRUCTION\nPlease convert the following context into an SPARQL query.\n\n### CONTEXT:\n{question}\n\n### SPARQL:\n{sparql}"
    return {'text': prompt}


# Command line arguments
parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--output_dir", default="/home/julio/repos/text2sparql/assets/models/KQAPRO_LCQUAD_OpenLLAMA_contrastive",
                    help="Output directory for the trained model")
parser.add_argument("--train_data", default="/home/julio/repos/text2sparql/data/kqapro_lcquad_train.json",
                    help="Output directory for the trained model")
parser.add_argument(
    "--model_id", default="openlm-research/open_llama_7b_v2", help="Pre-trained model ID")
parser.add_argument(
    "--max_steps", default=2000, help="number if taining steps")
parser.add_argument("--device_map", default='auto',
                    type=str, help="Device map for model.")


args = parser.parse_args()

output_dir = args.output_dir
model_id = args.model_id

# generator = SPARQLQueryGenerator()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

if args.device_map == 'auto':
    device_map = 'auto'
else:
    device_map = {"": 0}

# base_model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     quantization_config=bnb_config,
#     device_map=args.device_map
# )
base_model = AutoPeftModelForCausalLM.from_pretrained(
    args.model_id, device_map=device_map)


tokenizer = LlamaTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
base_model.resize_token_embeddings(len(tokenizer))
print(base_model)




train_data = load_data_from_file(args.train_data)
train_dataset = Dataset.from_dict(train_data)
# dataset_dict = DatasetDict({"train": train_dataset, "test": []})

mapped_datasets = train_dataset.map(generate_prompt)
print(mapped_datasets[0])

qlora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

supervised_finetuning_trainer = SFTTrainer(
    base_model,
    train_dataset=mapped_datasets,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        max_steps=int(args.max_steps),
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        fp16=True,
    ),
    tokenizer=tokenizer,
    peft_config=qlora_config,
    dataset_text_field="text",
    max_seq_length=1024
)

# Train
supervised_finetuning_trainer.train()
