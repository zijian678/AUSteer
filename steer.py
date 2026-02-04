import torch
import pyvene as pv
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModel, LlamaTokenizer, LlamaModel,AutoModelForCausalLM
import random

def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped

class Collector():
    collect_state = True
    collect_action = False
    def __init__(self, multiplier, head):
        self.head = head
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, b, s):
        if self.head == -1:
            # print('b:', b.shape)
            # self.states.append(b[0, -1].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
            self.states.append(b.detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
            # print('b2:', b.shape)
        else:
            # print('b:',b.shape)
            self.states.append(b[0, -1].reshape(32, -1)[self.head].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
            # print('b:', b.shape)
        return b

def get_llama_activations_pyvene(collected_model, collectors, prompt, device):
    with torch.no_grad():
        prompt = prompt.to(device)
        output = collected_model({"input_ids": prompt, "output_hidden_states": True})[1]
    hidden_states = output.hidden_states
    hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
    hidden_states = hidden_states.detach().cpu()#.numpy()
    head_wise_hidden_states = []
    for collector in collectors:
        if collector.collect_state:
            states_per_gen = torch.stack(collector.states, axis=0).to(torch.float16).cpu().numpy()
            head_wise_hidden_states.append(states_per_gen)
        else:
            head_wise_hidden_states.append(None)
        collector.reset()
    mlp_wise_hidden_states = []
    head_wise_hidden_states = torch.stack([torch.tensor(h) for h in head_wise_hidden_states], dim=0).squeeze()#.numpy()
    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states

def get_llama_activations_pyvene2(collected_model, collectors, input_ids, device):
    """
    Args:
        collected_model: model wrapped with pyvene collectors, callable like a HF model
        collectors: iterable of pyvene collectors. Each collector should have:
            - .collect_state (bool)
            - .states (list[Tensor]) collected during forward
            - .reset() method
            - attributes to identify layer/kind (any of: .layer_idx or .layer_id;
              and .kind or .name or .module_name including 'attn'/'attention' or 'mlp')
        input_ids: LongTensor [batch, seq]
        device: torch.device

    Returns:
        hidden_states: Tensor [n_layers+1, batch, seq, hidden]
        attn_outputs_by_layer: list length n_layers; each is Tensor [n_heads, batch, seq, hidden] or None
        mlp_outputs_by_layer:  list length n_layers; each is Tensor [batch, seq, hidden] or stacked [k, batch, seq, hidden] or None
    """
    from collections import defaultdict
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = collected_model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True
        )

    # 1) All layer hidden states (includes embedding output at index 0)
    # outputs.hidden_states is a tuple(len=n_layers+1) of [batch, seq, hidden]
    hidden_states = torch.stack(
        [hs.detach().to('cpu') for hs in outputs.hidden_states], dim=0
    )  # [n_layers+1, B, S, H]

    # 2) Parse collectors into attention vs MLP, grouped by layer
    attn_by_layer = defaultdict(list)  # layer_idx -> list of tensors [B, S, H]
    mlp_by_layer  = defaultdict(list)  # layer_idx -> list of tensors [B, S, H]

    def _get_layer_idx(c):
        return getattr(c, 'layer_idx', getattr(c, 'layer_id', None))

    def _get_kind(c):
        s = (getattr(c, 'kind', None) or
             getattr(c, 'name', None) or
             getattr(c, 'module_name', '') or '')
        s = str(s).lower()
        if 'attn' in s or 'attention' in s:
            return 'attn'
        if 'mlp' in s or 'feedforward' in s or 'ffn' in s:
            return 'mlp'
        return 'other'

    for c in collectors:
        try:
            if not getattr(c, 'collect_state', False):
                continue

            # c.states is typically a list (one per forward); stack and squeeze the "num_forwards" dim.
            if len(c.states) == 0:
                continue

            stacked = torch.stack(
                [t.detach().to('cpu') for t in c.states], dim=0
            )  # [num_forwards, B, S, H] or similar
            # If we only ran one forward pass, drop that leading dim
            if stacked.shape[0] == 1:
                stacked = stacked[0]  # [B, S, H]

            layer_idx = _get_layer_idx(c)
            kind = _get_kind(c)

            if layer_idx is None:
                # If no layer index is available, skip safely.
                continue

            if kind == 'attn':
                attn_by_layer[layer_idx].append(stacked)  # one per head/module
            elif kind == 'mlp':
                mlp_by_layer[layer_idx].append(stacked)
        finally:
            # Always reset collectors after reading
            if hasattr(c, 'reset'):
                c.reset()

    # 3) Build per-layer tensors
    # Determine n_layers from hidden_states (n_layers+1 on dim 0)
    n_layers = hidden_states.shape[0] - 1

    attn_outputs_by_layer = []
    for layer in range(n_layers):
        if layer in attn_by_layer and len(attn_by_layer[layer]) > 0:
            # Stack heads/modules along a new 0-dim => [n_heads, B, S, H]
            layer_attn = torch.stack(attn_by_layer[layer], dim=0)
            attn_outputs_by_layer.append(layer_attn)
        else:
            attn_outputs_by_layer.append(None)

    mlp_outputs_by_layer = []
    for layer in range(n_layers):
        if layer in mlp_by_layer and len(mlp_by_layer[layer]) > 0:
            # If multiple mlp collectors exist (uncommon), stack them on 0-dim: [k, B, S, H]
            # If only one, squeeze to [B, S, H]
            collected = mlp_by_layer[layer]
            if len(collected) == 1:
                mlp_outputs_by_layer.append(collected[0])
            else:
                mlp_outputs_by_layer.append(torch.stack(collected, dim=0))
        else:
            mlp_outputs_by_layer.append(None)

    return hidden_states, attn_outputs_by_layer, mlp_outputs_by_layer

def obtain_act(model, tokenizer, samples,applied_module):
    layer_num = model.config.num_hidden_layers
    head_num = model.config.num_attention_heads
    head_dim = model.config.head_dim
    collectors = []
    pv_config = []
    for layer in range(model.config.num_hidden_layers):
        collector = Collector(multiplier=0,
                              head=-1)  # head=-1 to collect all head activations, multiplier doens't matter
        collectors.append(collector)
        if applied_module == 'attention':
            pv_config.append({
                "component": f"model.layers[{layer}].self_attn.o_proj.input",
                "intervention": wrapper(collector),
            })

        if applied_module == 'ffn':
            pv_config.append({
                "component": f"model.layers[{layer}].mlp.output",
                "intervention": wrapper(collector),
            })

        if applied_module == 'layer':
            pv_config.append({
                "component": f"model.layers[{layer}].output",
                "intervention": wrapper(collector),
            })
        # pv_config.append({
        #     "component": f"model.layers[{layer}].self_attn.v_proj.output",
        #     "intervention": wrapper(collector),
        # })

    ###added
    # from rmodels.modeling_gemma2 import Gemma2ForCausalLM
    # model = Gemma2ForCausalLM.custom_from_pretrained(
    #     pretrained_model_name_or_path='google/gemma-2-2b-it',
    #     # cache_dir='/home/ntu3/.cache/huggingface/hub',
    #     torch_dtype=torch.float16,
    #     applied_module=applied_module
    # )
    # from utils.utils import zero_init
    # zero_init(model)
    ## edn
    collected_model = pv.IntervenableModel(pv_config, model)
    sample_acts = []
    for s in tqdm(samples):
        sen_ids = tokenizer(
            s,
            return_tensors='pt',
            # padding='max_length',  # Pads sequences to max_length
            truncation=True,  # Truncates sequences longer than max_length
            max_length=512  # Sets fixed sequence length
        ).input_ids

        layer_wise_activations, target_activations, _ = get_llama_activations_pyvene(collected_model, collectors, sen_ids,
                                                                                    device=model.device)

        # hidden_states, attn_outputs_by_layer, mlp_outputs_by_layer = get_llama_activations_pyvene2(model, collectors, sen_ids, device=model.device)
        # print('hidden_states:',hidden_states.shape,hidden_states)
        # print('attn_outputs_by_layer:',attn_outputs_by_layer.shape,attn_outputs_by_layer)
        # print('mlp_outputs_by_layer:',mlp_outputs_by_layer.shape,mlp_outputs_by_layer)
        # print('target_activations:',target_activations.shape,target_activations)
        # print('layer_wise_activations:',layer_wise_activations.shape,layer_wise_activations)
        # layer_wise_activations: (33, 145, 4096) head_wise_activations: (32, 145, 4096)
        # print('sen_ids:',sen_ids.shape,tokenizer.convert_ids_to_tokens(sen_ids[0]))
        # print('layer_wise_activations:',layer_wise_activations.shape,'head_wise_activations:',head_wise_activations.shape)
        current_act = target_activations[:,-1,:]
        # current_act = current_act.reshape(layer_num,head_num,head_dim)
        # current_act = current_act.reshape(layer_num * head_num,head_dim) # torch.Size([1024, 128])
        # print('current_act:',current_act.shape)
        sample_acts.append(current_act)
    sample_acts = torch.stack(sample_acts)
    return sample_acts

def sliding_window_indices(N, c,overlap=False):
    #N: total length
    #c: window size
    indices = []
    if overlap == True:
        return [(i, i + c) for i in range(N - c + 1)]
    else:
        return [(i, min(i + c , N )) for i in range(0, N, c)]

def MFU_screening(train_dataset,model, tokenizer,path,window_size = 1,applied_module=None):
    layer_num = model.config.num_hidden_layers
    head_num = model.config.num_attention_heads
    head_dim = model.config.head_dim
    hidden_dim = model.config.hidden_size
    print('model config:','layer_num',layer_num,'head_num',head_num,'head_dim',head_dim,'hidden_dim',hidden_dim)

    samples = train_dataset
    print('# of probing samples:',len(samples),samples[:1])
    pos_samples = []
    for td in train_dataset:
        if "I'm sorry, but I can't assist" in td[1]:
            # positive_words = [
            #     "happy", "joyful", "bright", "optimistic", "friendly", "kind", "loving", "cheerful",
            #     "peaceful", "grateful", "brilliant", "successful", "amazing", "fantastic", "wonderful",
            #     "generous", "thoughtful", "hopeful", "smiling", "sunny", "playful", "creative", "fun",
            #     "inspiring", "motivated", "energetic", "calm", "caring", "compassionate", "courageous",
            #     "determined", "enthusiastic", "forgiving", "gentle", "genuine", "glowing", "helpful",
            #     "honest", "humble", "imaginative", "innovative", "jolly", "kindhearted", "lighthearted",
            #     "loyal", "mindful", "noble", "open-minded", "outgoing", "patient", "positive", "radiant",
            #     "reliable", "resilient", "respectful", "selfless", "sincere", "spirited", "strong",
            #     "supportive", "talented", "thankful", "trustworthy", "uplifting", "vibrant", "warm",
            #     "wise", "adventurous", "affectionate", "ambitious", "appreciative", "balanced", "bold",
            #     "bubbly", "charming", "confident", "considerate", "cooperative", "courageous", "dedicated",
            #     "delightful", "devoted", "dynamic", "encouraging", "faithful", "fearless", "flexible",
            #     "friendly", "generous", "genuine", "happy-go-lucky", "hardworking", "helpful", "honorable",
            #     "humorous", "influential", "jovial"
            # ]
            # selected_words = random.sample(positive_words, 50)
            # pos_sen = ' '.join(selected_words)
            # # pos_samples.append(td[0] + td[1])
            pos_samples.append(td[1])
            # pos_samples.append(pos_sen)

            # pos_samples.append(pos_sen)
        else:
            pos_samples.append(td[0] + td[1])
    neg_samples = []
    for td in train_dataset:
        neg_samples.append(td[0] + td[2])
        # neg_samples.append('   ')
    print('pos_samples:',len(pos_samples),pos_samples[:1])
    print('neg_samples:',len(neg_samples),neg_samples[:1])

    model.eval()
    pos_acts = obtain_act(model, tokenizer, pos_samples,applied_module)
    neg_acts = obtain_act(model, tokenizer, neg_samples,applied_module)
    # print('pos_acts:', pos_acts.shape,'neg_acts:',neg_acts.shape) # pos_acts: torch.Size([10, 36, 4096]) neg_acts: torch.Size([10, 36, 4096])

    #saved_results
    all_layer_ids = []
    all_indices = []
    all_scores = []
    # all_coeffs = []

    for lid in range(layer_num):
        pos_activations = pos_acts[:,lid,:] # 1000 * 4096
        neg_activations = neg_acts[:,lid,:]

        if applied_module == 'attention':
            all_wins = sliding_window_indices(head_dim * head_num, window_size)
        else:
            all_wins = sliding_window_indices(hidden_dim, window_size)
        all_gains = []
        for current_ids in tqdm(all_wins):
        # current_ids = [64,96]
            pos_activations_truncted = pos_activations[:,current_ids[0]:current_ids[1]] # activations_truncted: torch.Size([1000, 32])
            neg_activations_truncted = neg_activations[:,current_ids[0]:current_ids[1]]

            logit_gains = pos_activations_truncted - neg_activations_truncted # logit_gains: torch.Size([1000, 40])

            # print('logit_gains:',logit_gains.shape) # logit_gains: torch.Size([1000, 40])
            larger0 = (logit_gains > 0).sum(dim=0)
            small0 = (logit_gains < 0).sum(dim=0)
            # print('larger0:',larger0.shape,larger0)
            # print('small0:',small0.shape,small0)
            cc = torch.max(larger0, small0)
            beta = cc/logit_gains.shape[0]
            if larger0 < small0:
                beta = -1.0 * beta
            # print('larger0:',larger0,'small0:',small0,'beta:',beta)
            # print('cc:',cc.shape,cc,'beta:',beta)
            # mean_biases = torch.mean(logit_gains,dim = 0)
            # print('mean_biases:',mean_biases.shape,mean_biases)
            # final_score = torch.sum(cc)
            # print('larger0:',larger0.shape,larger0,'small0:',small0.shape,small0,'cc:',cc.shape,cc,'final_score:',final_score)
            # mean_gains = final_score

            # mean_gains = torch.mean(logit_gains,dim = 0)#/(torch.mean(torch.abs(pos_activations_truncted))
            #                                                  #+torch.mean(torch.abs(neg_activations_truncted)))
            # # print('mean_gains:',mean_gains.shape)
            # mean_gains = torch.sum(torch.abs(mean_gains))
            # print('mean_gains22:',mean_gains)

            ### save results
            all_scores.append(float(beta))
            # all_coeffs.append(mean_biases)
            all_layer_ids.append(lid)
            all_indices.append(current_ids)

    torch.save(all_layer_ids, path + f'/all_layer_ids.pt')
    torch.save(all_indices, path + f'/all_indices.pt')
    torch.save(all_scores, path + f'/all_scores.pt')
    return all_scores,all_layer_ids,all_indices
            # all_scores.append(float(mean_gains))
        # file_name = f'layer_{str(lid)}_win_{str(window_size)}_overlap_{str(overlap)}.csv'
        # np.savetxt(file_name, all_gains, delimiter=",")
    # print('all_layer_ids:',len(all_layer_ids),all_layer_ids)
    # print('all_indices:',len(all_indices),all_indices)
    # print('all_scores:',len(all_scores),all_scores)
    # # all_coeffs = torch.stack(all_coeffs)
    # print('all_coeffs:',len(all_coeffs))
    # file_name = f'win_{str(window_size)}_overlap_{str(overlap)}'
    # os.makedirs(file_name, exist_ok=True)
    # path = f'./{file_name}/'
    # torch.save(all_layer_ids, path + f'all_layer_ids.pt')
    # torch.save(all_indices, path + f'all_indices.pt')
    # torch.save(all_scores, path + f'all_scores.pt')
    # torch.save(all_coeffs, path + f'all_coeffs.pt')
def set_MFU(all_scores,all_layer,all_indices,k,alpha,model,applied_module):
    non_zero_elements = k * 1
    all_scores_abs = [abs(gg) for gg in all_scores]
    sorted_indices = sorted(range(len(all_scores_abs)), key=lambda i: all_scores_abs[i], reverse=True)
    sorted_values = [all_scores[i] for i in sorted_indices]
    # print('sorted_values:', sorted_values[:10])
    # print('sorted_indices:', sorted_indices[:10])

    layer_num = model.config.num_hidden_layers
    head_num = model.config.num_attention_heads
    head_dim = model.config.head_dim
    hidden_dim = model.config.hidden_size

    if applied_module == 'attention':
        mask = torch.zeros(layer_num, head_num*head_dim)
    else:
        mask = torch.zeros(layer_num, hidden_dim)



    for i in sorted_indices:
        lid = all_layer[i]
        ids = all_indices[i]
        beta = all_scores[i]
        # print('i:',i,'lid:',lid,'ids:',ids,'beta:',beta)
        # print('mask[lid,ids[0]:ids[1]]:',mask[lid,ids[0]:ids[1]].shape)
        # print('mean bias:',all_coeffs[i])

        # static scaling
        mask[lid, ids[0]:ids[1]] = 1.0 * beta

        # adaptive scaling
        # mask[lid, ids[0]:ids[1]] = all_coeffs[i]

        condition_sum = torch.count_nonzero(mask)
        # print('condition_sum:',condition_sum)

        if condition_sum >= non_zero_elements:
            break
    # print('mask:',mask.shape,mask)

    mask = mask * alpha

    # self.layers[idx].applied_module = applied_module
    # self.layers[idx].self_attn.applied_module = applied_module
    # self.layers[idx].mlp.applied_module = applied_module

    if applied_module == "attention":
        for jj in range(model.config.num_hidden_layers):
            model.model.layers[jj].self_attn.activation_mask.data.copy_(mask[jj])
    if applied_module == "ffn":
        for jj in range(model.config.num_hidden_layers):
            model.model.layers[jj].mlp.activation_mask.data.copy_(mask[jj])
    if applied_module == "layer":
        for jj in range(model.config.num_hidden_layers):
            model.model.layers[jj].activation_mask.data.copy_(mask[jj])
