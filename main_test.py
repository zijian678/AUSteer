import os
os.environ["HF_HOME"] = "/mnt/data/ntu_volume/act/cache"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import argparse
from utils.load_dataset import *
from tqdm import tqdm
from rmodels.modeling_qwen3 import Qwen3ForCausalLM
from rmodels.modeling_llama import LlamaForCausalLM
from rmodels.modeling_gemma2 import Gemma2ForCausalLM
from utils.utils import zero_init, get_model_answer,extract_gsm8k_number,get_model_gsm8k,get_model_svamp
from steer import MFU_screening,set_MFU
from time import sleep
from transformers import logging
logging.set_verbosity_error()   # or: logging.set_verbosity(logging.ERROR)


parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=5.0)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--path', type=str, default='./AU_ranks/')
parser.add_argument('--steer', type=bool, default=True)
parser.add_argument('--model', type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument('--data_name', type=str, default="copa")
parser.add_argument('--applied_module', type=str, default="ffn",help = ['attention','ffn','layer'])
parser.add_argument('--device', type=str, default="cuda:3")
args = parser.parse_args()

# Qwen/Qwen3-8B
# meta-llama/Llama-2-7b-chat-hf
#google/gemma-3-1b-it
# google/gemma-2-9b-it
# meta-llama/Llama-2-7b-chat-hf
#winogrande
#meta-llama/Meta-Llama-3-8B


model_name = args.model
alpha = args.alpha
k = args.k
path = args.path
steer = args.steer
applied_module = args.applied_module
data_name = args.data_name

if "Qwen3" in model_name:
    model = Qwen3ForCausalLM.custom_from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype=torch.float16,
        applied_module=applied_module
    )
elif "llama" in model_name:
    model = LlamaForCausalLM.custom_from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype=torch.float16,
        applied_module=applied_module
    )
elif "gemma" in model_name:
    model = Gemma2ForCausalLM.custom_from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype=torch.float16,
        applied_module=applied_module
    )
# print('model device:',model.device)
model = model.to(args.device)
print('model device22:',model.device)


for name, param in model.named_parameters():
    print(name, param.device)
    break  # 去掉 break 可以全部打印

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

print('before',model.model.layers[5].self_attn.activation_mask)
zero_init(model)

print('after:',model.model.layers[5].self_attn.activation_mask)

data_collect = {
    'boolq':load_boolq,
    'copa':load_copa,
    'storycloze':load_storycloze,
    'winogrande':load_winogrande,
    'gsm8k':load_gsm8k,
    'svamp':load_svamp,
    'mawps':load_mawps,
    'toxic':load_toxic,
    'bpo':load_bpo
}

data_generate_length = {
    'boolq':1,
    'copa':1,
    'storycloze':1,
    'winogrande':1,
    'gsm8k':400,
    'svamp':5,
    'mawps':250, # 150 for qwen3
    'toxic':50,
    'bpo':500
}
if 'Qwen' in model_name:
    data_generate_length['svamp'] = 100
    data_generate_length['mawps'] = 150

if 'Llama' in model_name:
    data_generate_length['svamp'] = 50

name_save = {
    "Qwen/Qwen3-8B":"qwen3_8b",
    "meta-llama/Llama-2-7b-chat-hf":"llama2_7b_chat",
    "google/gemma-2-2b-it":"gemma2_2b_it",
    "google/gemma-2-9b-it":"gemma2_9b_it",
    "meta-llama/Llama-2-13b-chat-hf":"llama2_13b_chat",
    "meta-llama/Llama-3.1-8B":"Llama_3_8B",
    "meta-llama/Meta-Llama-3-8B":"meta_llama_3_8B",
    "meta-llama/Llama-3.1-8B-Instruct":"meta_llama_3_8B_Instruct",
}

train_dataset, test_dataset = data_collect[data_name]()
print(f'train sample: {len(train_dataset)}, test sample: {len(test_dataset)}')

if steer:
    # obtain AU ranks -- MFU means AU
    MFU_path = path + name_save[model_name] + f'_{data_name}_{applied_module}'
    if os.path.exists(MFU_path + '/all_layer_ids.pt'):
        all_layer_ids = torch.load(MFU_path + f'/all_layer_ids.pt')
        all_indices = torch.load(MFU_path + f'/all_indices.pt')
        all_scores = torch.load(MFU_path + f'/all_scores.pt')
    else:
        os.makedirs(MFU_path,exist_ok=True)
        all_scores,all_layer_ids,all_indices = MFU_screening(train_dataset[:1000], model, tokenizer,MFU_path, window_size=1,applied_module=applied_module)

    # set
    set_MFU(all_scores, all_layer_ids, all_indices, k, alpha, model,applied_module)




# evaluation
if data_name == 'gsm8k' or data_name == 'mawps':
    total = 0
    correct = 0
    for example in tqdm(test_dataset):
        pred = get_model_gsm8k(example[0], model, tokenizer, max_new_tokens=data_generate_length[data_name])
        # print('pred sentence:',pred.lower())
        pred_number = extract_gsm8k_number(pred)
        correct_number = extract_gsm8k_number(example[1])

        try:
            pred_number = float(pred_number)
        except:
            pred_number = None
        # print('pred:',pred_number,'correct_number:',float(correct_number))
        if pred_number == float(correct_number):
            correct += 1
    Accuracy = correct / len(test_dataset)
    print(f'evaluation results on {model_name}: dataset {data_name} alpha {alpha} k {k} Accuracy {Accuracy}')
elif data_name == 'svamp':
    total = 0
    correct = 0
    for example in tqdm(test_dataset):
        pred = get_model_svamp(example[0], model, tokenizer, max_new_tokens=data_generate_length[data_name])
        # print('pred sentence:',pred.lower())
        pred_number = extract_gsm8k_number(pred)
        correct_number = extract_gsm8k_number(example[1])

        try:
            pred_number = float(pred_number)
        except:
            pred_number = None
        # print('pred:',pred_number,'correct_number:',float(correct_number))
        if pred_number == float(correct_number):
            correct += 1
    Accuracy = correct / len(test_dataset)
    print(f'evaluation results on {model_name}: dataset {data_name} alpha {alpha} k {k} Accuracy {Accuracy}')

else:
    total = 0
    correct = 0
    for example in tqdm(test_dataset):
        pred = get_model_answer(example[0],model, tokenizer,max_new_tokens=data_generate_length[data_name])
        # print('pred:',pred.lower(),'ground-truth:',example[1].strip().lower())
        if pred.lower() == example[1].strip().lower():
            correct += 1
    Accuracy = correct / len(test_dataset)
    print(f'evaluation results on {model_name}: dataset {data_name} module {applied_module} alpha {alpha} k {k} Accuracy {Accuracy}')
