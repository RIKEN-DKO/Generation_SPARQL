#%%
DEBUG=False

# %load_ext autoreload
# %autoreload 2
#%%
import argparse
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import LlamaTokenizer
import bitsandbytes as bnb
import torch
import os
import json
import datetime
import transformers
from tqdm import tqdm
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from trl import SFTTrainer
from text2sparql.format import truncate_batch, extract_first_sparql, load_data_from_file, generate_prompt, prompt_question, prompt_question_contrastivev1, prompt_append_BERN2EL,prompt_append_SpacyWikidataEL,extract_first_sparql_contrastive,extract_first_sparql_bgee
from text2sparql.metrics import eval_pairs
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from peft import AutoPeftModelForCausalLM
import random

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




#%%
def make_inference(question):
    batch = tokenizer(prompt_question(question), return_tensors='pt')
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(
            input_ids=batch["input_ids"].to("cuda:0"), max_new_tokens=2048
        )
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

#%%
def batched_inference(questions, batch_size=8, max_new_tokens=512,prompt_type='plain',nlp=None):
    results = []
    args = {}
    if prompt_type in ['plain', 'plainbgee']:
        prompt = prompt_question
    elif prompt_type == 'contrastive':
        prompt = prompt_question_contrastivev1
    elif prompt_type == 'bern2el':
        prompt = prompt_append_BERN2EL
    elif prompt_type == 'wikidatael':
        prompt = prompt_append_SpacyWikidataEL
        args = {'nlp':nlp}
    elif prompt_type == 'wikidataelcon':
        prompt = prompt_append_SpacyWikidataEL
        args = {'nlp':nlp}
        
    # Create an iterable with tqdm
    iterable = range(0, len(questions), batch_size)
    iterable = tqdm(iterable, total=len(iterable), desc="Processing batches")

    for i in iterable:
        batch_questions = [prompt(q,**args)
                           for q in questions[i:i + batch_size]]
        batch = tokenizer(batch_questions, return_tensors='pt',
                          padding=True, truncation=True,)
        # batch = truncate_batch(batch)
        with torch.cuda.amp.autocast():
            output_tokens = model.generate(
                input_ids=batch["input_ids"].cuda(), max_new_tokens=max_new_tokens,
                do_sample=False
            )
        batch_out = [tokenizer.decode(
            tokens, skip_special_tokens=True) for tokens in output_tokens]
        results.extend(batch_out)

    return results

#%%
parser = argparse.ArgumentParser(description="Inference Script")
parser.add_argument("--batch_size", default=20, type=int,
                    help="Batch size for inference.")
parser.add_argument("--chunk", default=10, type=int,
                    help="Batch size for inference.")
parser.add_argument("--max_new_tokens", default=1024, type=int,
                    help="model lenght size")
parser.add_argument("--device_map", default='auto',
                    type=str, help="Device map for model.")
parser.add_argument(
    "--output_dir", default="/home/julio/repos/text2sparql/assets/models/BGEE_meaningful_vars_comments_augmented/checkpoint-2000", help="Path to model checkpoint.")
parser.add_argument(
    "--dataset", default='/home/julio/repos/text2sparql/data/bgee_augment/meaningful_vars_comments_augmented_test.json', help="Path to dataset.")
parser.add_argument(
    "--model_id", default="", help="Model ID for loading.")# DEBUG openlm-research/open_llama_7b_v2
parser.add_argument(
    "--prompt_type", default="plainbgee", choices=['plain', 'contrastive', 'bern2el', 'wikidatael', 'wikidataelcon','plainbgee'])


## DEBUG
if DEBUG:
    args = parser.parse_args('')
else:
    args = parser.parse_args()
#%%
if args.device_map == 'auto':
    device_map = 'auto'
else:
    device_map = {"": 0}

test_data = load_data_from_file(args.dataset)
test_dataset = Dataset.from_dict(test_data)
# print(test_dataset['question'])
print(f"Size of test_dataset: {test_dataset.num_rows} examples")

if args.model_id != "":
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_path = args.model_id
    model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map=args.device_map
    )
else:
    model_path = args.output_dir
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path, device_map=device_map, torch_dtype=torch.bfloat16)
#%%
print('model:',model_path)    
tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))


# print(model)

test_data = load_data_from_file(args.dataset)
test_dataset = Dataset.from_dict(test_data)
print(test_dataset[0])
print(f"Size of test_dataset: {test_dataset.num_rows} examples")

#%% test
# args.prompt_type = 'contrastive'
# predicted_sparql = batched_inference(test_dataset['question'][:1], batch_size=1,
#                   max_new_tokens=256, prompt_type=args.prompt_type)
# clean_pred_sparql = [extract_first_sparql(spa) for spa in predicted_sparql]

#test

#
if args.prompt_type in ['wikidatael', 'wikidataelcon']:
    #load the spacy nlp
    import spacy
    nlp = spacy.blank("en")
    nlp.add_pipe('opentapioca')
else: 
    nlp = None

#%%
if args.chunk > test_dataset.num_rows:
    chunk = None    
else:
    chunk = args.chunk
    print('But only evaluating on', chunk)

predicted_sparql = batched_inference(
    test_dataset['question'][:chunk], batch_size=args.batch_size,max_new_tokens=args.max_new_tokens,
    prompt_type=args.prompt_type,nlp=nlp)


#%% DEBUG
if DEBUG:
    for i,a in enumerate(predicted_sparql):
        print(i)
        print(a)
        print('--')
#%%
if args.prompt_type in ['contrastive', 'wikidataelcon']:
    extractor = prompt_question_contrastivev1
if args.prompt_type in ['plainbgee']:
    extractor = extract_first_sparql_bgee
else:
    extractor = extract_first_sparql
# print(predicted_sparql)
clean_pred_sparql = [extractor(spa) for spa in predicted_sparql]
# print(clean_pred_sparql)
#%% DEBUG
if DEBUG:
    for i,e in enumerate(clean_pred_sparql):
        print(i)
        print(e)
        print('=========')
    
#%%
results = eval_pairs(
    zip(test_dataset['sparql'][:chunk], clean_pred_sparql))

# ...

# Generate the new directory name based on date and time
current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
eval_dir = os.path.join(args.output_dir, f'eval_{current_time}')

# Create the directory if it doesn't exist
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

# Inform the user where the files will be saved
print(
    f"Saving evaluation results and arguments to: {os.path.join(eval_dir, 'eval.json')}")

# Save results and arguments to eval.json
with open(os.path.join(eval_dir, 'eval.json'), 'w') as f:
    json.dump({
        'results': results,
        'arguments': vars(args)
    }, f)


# Inform the user where the pairs file will be saved
print(f"Saving pairs to: {os.path.join(eval_dir, 'pairs.json')}")

# Save pairs to pairs.json
with open(os.path.join(eval_dir, 'pairs.json'), 'w') as f:
    json.dump(
        list(zip(test_dataset['sparql'][:chunk], clean_pred_sparql)), f)

# %%
handle = open(os.path.join(eval_dir, 'predicted.txt'), 'w')
for i,a in enumerate(predicted_sparql):
    handle.write(str(i)+':\n')
    handle.write(a)
    handle.write('\n\n--\n\n')
handle.close()

handle = open(os.path.join(eval_dir, 'extracted.txt'), 'w')
for i,a in enumerate(clean_pred_sparql):
    if a is not None:
        handle.write(str(i)+':\n')
        handle.write(a)
        handle.write('\n\n--\n\n')
handle.close()