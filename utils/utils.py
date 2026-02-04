import torch

def zero_init(model):
    print('model.config.num_hidden_layers:',model.config.num_hidden_layers)
    for i in range(model.config.num_hidden_layers):
            attn_v = model.model.layers[i].self_attn.activation_mask
            for j, module in enumerate(attn_v):
                module.data.zero_()
            ffn_v = model.model.layers[i].mlp.activation_mask
            for j, module in enumerate(ffn_v):
                module.data.zero_()
            layer_v = model.model.layers[i].activation_mask
            for j, module in enumerate(layer_v):
                module.data.zero_()

            # self.layers[idx].applied_module = applied_module
            # self.layers[idx].self_attn.applied_module = applied_module
            # self.layers[idx].mlp.applied_module = applied_module
def zero_init_moe(model):
    print('model.config.num_hidden_layers:',model.config.num_hidden_layers)
    for i in range(model.config.num_hidden_layers):
            attn_v = model.model.layers[i].self_attn.activation_mask
            for j, module in enumerate(attn_v):
                module.data.zero_()
            # ffn_v = model.model.layers[i].mlp.activation_mask
            # for j, module in enumerate(ffn_v):
            #     module.data.zero_()
            # layer_v = model.model.layers[i].activation_mask
            # for j, module in enumerate(layer_v):
            #     module.data.zero_()

def extract_gsm8k_number(text: str):
    import re
    # 1) Try the "#### final answer" convention
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?|\d+\s*/\s*\d+)", text)
    if m:
        return m.group(1).replace(" ", "")
    # 2) Else take the last number or simple fraction in the whole output
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?|\d+\s*/\s*\d+", text)
    return nums[-1].replace(" ", "") if nums else None

def extract_svamp(text: str):
    import re
    # 1) Try the "#### final answer" convention
    m = re.search(r"Answer:\s*([-+]?\d+(?:\.\d+)?|\d+\s*/\s*\d+)", text)
    if m:
        return m.group(1).replace(" ", "")
    # 2) Else take the last number or simple fraction in the whole output
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?|\d+\s*/\s*\d+", text)
    return nums[-1].replace(" ", "") if nums else None

def get_model_answer(prompt, model, tokenizer,max_new_tokens=1):
    # max_new_tokens = 3
    # print('model.device:',model.device)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    # print('inputs:',inputs)
    # print('outputs:',outputs)

    # print(tokenizer.convert_ids_to_tokens(outputs[0]))
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # print('prompt:',prompt)
    # print('\ndecoded:',decoded)
    answer = decoded[len(prompt):].strip()

    # print('answer:',answer)
    return answer


def get_model_gsm8k(prompt, model, tokenizer, max_new_tokens=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # print('prompt:',prompt)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    # print(tokenizer.convert_ids_to_tokens(outputs[0]))
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)



    # print('prompt:',prompt)
    # print('\ndecoded:',decoded)
    answer = decoded[len(prompt):].strip()

    # print('******decoded:', answer)

    trigger = "####"
    idx = answer.find(trigger)
    if idx != -1:
        # Split into tokens so we can count
        tokens = answer[idx:].split()
        # Keep "####" plus 5 tokens after
        cutoff = " ".join(tokens[:1 + 1])  # 1 for "####" + 5 more
        answer = answer[:idx] + cutoff
    # print('******refined decoded:', answer)

    # print('answer:',answer)
    return answer

def get_model_svamp(prompt, model, tokenizer, max_new_tokens=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    # print(tokenizer.convert_ids_to_tokens(outputs[0]))
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)



    # print('prompt:',prompt)
    # print('\ndecoded:',decoded)
    answer = decoded[len(prompt):].strip()

    # print('******decoded:', answer)

    trigger = "Answer"
    idx = answer.find(trigger)
    if idx != -1:
        # Split into tokens so we can count
        tokens = answer[idx:].split()
        # Keep "####" plus 5 tokens after
        cutoff = " ".join(tokens[:1 + 1])  # 1 for "####" + 5 more
        answer = answer[:idx] + cutoff
    # print('******refined decoded:', answer)

    # print('answer:',answer)
    return answer

def get_toxic(prompt, model, tokenizer, max_new_tokens=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            # temperature = 2.0
        )

    # print(tokenizer.convert_ids_to_tokens(outputs[0]))
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)



    # print('prompt:',prompt)
    # print('\ndecoded:',decoded)
    answer = decoded[len(prompt):].strip()

    # print('******decoded:', answer)

    # print('answer:',answer)
    return answer

def general_construct(samples):
    pos_samples = []
    neg_samples = []
    for i in samples:
        pos_samples.append(i[0] + i[1])
        neg_samples.append(i[0] + i[2])
    return pos_samples,neg_samples

def obtain_mfu_score(dataset_name, train_dataset,model,tokenizer):
    samples = train_dataset
    print('# of probing samples:', len(samples), samples[:1])

