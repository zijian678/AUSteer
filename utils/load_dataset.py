import json
import pandas as pd
import random
import csv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

def load_storycloze():
    train_data, test_data = [], []
    with open(f"./datasets/StoryCloze/spring2016.val.en.tsv.split_20_80_train.tsv", "r", encoding="utf-8") as g:
        lines = g.readlines()
    lines = lines[1:]
    for ind in range(len(lines)):
        line = lines[ind].strip()
        line = line.split('\t')
        sent1, sent2, sent3, sent4, quiz1, quiz2, label = line[1], line[2], line[3], line[4], line[5], line[6], line[7]
        sents = sent1 + ' ' + sent2 + ' ' + sent3 + ' ' + sent4
        prompt = f'{sents}\nQuestion: What is a possible continuation for the story given the following options?\nA: {quiz1} B: {quiz2}\nAnswer:'
        if int(label) == 1:
            cor_answer, wro_answer = 'A', 'B'
        elif int(label) == 2:
            cor_answer, wro_answer = 'B', 'A'
        train_data.append([prompt, cor_answer, wro_answer])

    with open(f"./datasets/StoryCloze/spring2016.val.en.tsv.split_20_80_eval.tsv", "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = lines[1:]
    for ind in range(len(lines)):
        line = lines[ind].strip()
        line = line.split('\t')
        sent1, sent2, sent3, sent4, quiz1, quiz2, label = line[1], line[2], line[3], line[4], line[5], line[6], line[7]
        sents = sent1 + ' ' + sent2 + ' ' + sent3 + ' ' + sent4
        prompt = f'{sents}\nQuestion: What is a possible continuation for the story given the following options?\nA: {quiz1} B: {quiz2}\nAnswer:'
        if int(label) == 1:
            cor_answer, wro_answer = 'A', 'B'
        elif int(label) == 2:
            cor_answer, wro_answer = 'B', 'A'
        test_data.append([prompt, cor_answer, wro_answer])
    print('data examples:', train_data[0], test_data[0])
    return train_data, test_data

def load_copa():
    train_data,test_data = [],[]
    with open(f"./datasets/COPA/train.csv", "r", encoding="utf-8") as g:
        lines = g.readlines()
    for line in lines[1:1501]:
        line = line.strip().split(',')
        label,premise,question,choice1,choice2 = line[0].strip(),line[2].strip(),line[3].strip(),line[4].strip(),line[5].strip()
        prompt = f'Question:\n{premise} Based on the previous passage, choose the most reasonable {question}.\nA:{choice1}\nB:{choice2}\n\nAnswer:\n'
        if int(label) == 0:
            cor_answer,wro_answer = 'A','B'
        elif int(label) == 1:
            cor_answer,wro_answer = 'B','A'
        train_data.append([prompt,cor_answer,wro_answer])

    with open(f"./datasets/COPA/test.csv", "r", encoding="utf-8") as g:
        lines = g.readlines()
    for line in lines[1:]:
        line = line.strip().split(',')
        label, premise, question, choice1, choice2 = line[0].strip(), line[2].strip(), line[3].strip(), line[4].strip(), \
        line[5].strip()
        prompt = f'Question:\n{premise} Based on the previous passage, choose the most reasonable {question}.\nA:{choice1}\nB:{choice2}\n\nAnswer:\n'
        if int(label) == 0:
            cor_answer, wro_answer = 'A', 'B'
        elif int(label) == 1:
            cor_answer, wro_answer = 'B', 'A'
        test_data.append([prompt, cor_answer, wro_answer])
    print('data examples:', train_data[0], test_data[0])
    return train_data,test_data

def load_copa22():
    train_data,test_data = [],[]
    with open(f"./datasets/COPA/train.csv", "r", encoding="utf-8") as g:
        lines = g.readlines()
    for line in lines[1:1501]:
        line = line.strip().split(',')
        label,premise,question,choice1,choice2 = line[0].strip(),line[2].strip(),line[3].strip(),line[4].strip(),line[5].strip()
        prompt = f'Answer the following question with A or B. Question:\n{premise} Based on the previous passage, choose the most reasonable {question}. Please only reply with A or B.\nA:{choice1}\nB:{choice2}\n\nAnswer:\n'
        if int(label) == 0:
            cor_answer,wro_answer = 'A','B'
        elif int(label) == 1:
            cor_answer,wro_answer = 'B','A'
        train_data.append([prompt,cor_answer,wro_answer])

    with open(f"./datasets/COPA/test.csv", "r", encoding="utf-8") as g:
        lines = g.readlines()
    for line in lines[1:]:
        line = line.strip().split(',')
        label, premise, question, choice1, choice2 = line[0].strip(), line[2].strip(), line[3].strip(), line[4].strip(), \
        line[5].strip()
        prompt = f'Answer the following question with A or B. Question:\n{premise} Based on the previous passage, choose the most reasonable {question}. Please only reply with A or B.\nA:{choice1}\nB:{choice2}\n\nAnswer:\n'
        if int(label) == 0:
            cor_answer, wro_answer = 'A', 'B'
        elif int(label) == 1:
            cor_answer, wro_answer = 'B', 'A'
        test_data.append([prompt, cor_answer, wro_answer])
    print('data examples:', train_data[0], test_data[0])
    return train_data,test_data


def load_boolq():
    train_data, test_data = [], []
    with open(f"./datasets/BoolQ/train.jsonl", "r", encoding="utf-8") as g:
        lines = g.readlines()
    lines = lines[:2000]
    for ind in range(len(lines)):
        line = json.loads(lines[ind])
        question, passage, answer = line['question'].strip(), line['passage'].strip(), str(line['answer']).strip()

        prompt = f"Is the answer to the question encapsulated in the passage? Please confirm with 'yes' or 'no'.\n\nPassage: {passage}\n\nQuestion: {question}\n\nAnswer:"
        if answer == 'True':
            cor_answer, wro_answer = 'Yes', 'No'
        elif answer == 'False':
            cor_answer, wro_answer = 'No', 'Yes'
        train_data.append([prompt, cor_answer, wro_answer])

    with open(f"./datasets/BoolQ/dev.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
    for ind in range(len(lines)):
        line = json.loads(lines[ind])
        question, passage, answer = line['question'].strip(), line['passage'].strip(), str(line['answer']).strip()

        prompt = f"Is the answer to the question encapsulated in the passage? Please confirm with 'yes' or 'no'.\n\nPassage: {passage}\n\nQuestion: {question}\n\nAnswer:"
        if answer == 'True':
            cor_answer, wro_answer = ' yes', ' no'
        elif answer == 'False':
            cor_answer, wro_answer = ' no', ' yes'
        test_data.append([prompt, cor_answer, wro_answer])
    print('data examples:',train_data[0], test_data[0])
    return train_data, test_data


def load_XNLI():
    train_data, test_data = [], []
    with open("./datasets/mnli/xnli.dev.tsv") as f:
        lines = f.readlines()
    copora = [[], [], [], [], [], [], [], [], [], [], [], [], []]
    for line in lines[1:]:
        line = line.split('\t')
        lang, label, premise, hypothesis = line[0], line[1], line[6], line[7]
        if lang == 'en':
            prompt = f"Question:\n{premise} Based on the previous passage, is it true that \"{hypothesis}\"? Please confirm with 'Yes', 'No', or 'Maybe'.\n\nAnswer:\n"
            prompt = f'Answer whether the hypothesis is more likely to be true, false, or unclusive based on the given premise.\nPremise: {premise}\nHypothesis: {hypothesis}\nAnswer:'
            if label == 'entailment':
                cor_answer, wro_answer = 'True', random.choice(['False', 'unclusive'])
            elif label == 'contradiction':
                cor_answer, wro_answer = 'False', random.choice(['True', 'unclusive'])
            elif label == 'neutral':
                cor_answer, wro_answer = 'unclusive', random.choice(['False', 'True'])
            train_data.append([prompt, cor_answer, wro_answer])

    with open("./datasets/mnli/xnli.test.tsv") as f:
        lines = f.readlines()
    copora = [[], [], [], [], [], [], [], [], [], [], [], [], []]
    for line in lines[1:]:
        line = line.split('\t')
        lang, label, premise, hypothesis = line[0], line[1], line[6], line[7]
        if lang == 'en':
            prompt = f'Answer whether the hypothesis is more likely to be true, false, or unclusive based on the given premise.\nPremise: {premise}\nHypothesis: {hypothesis}\nAnswer:'
            if label == 'entailment':
                cor_answer, wro_answer = 'True', random.choice(['False', 'unclusive'])
            elif label == 'contradiction':
                cor_answer, wro_answer = 'False', random.choice(['True', 'unclusive'])
            elif label == 'neutral':
                cor_answer, wro_answer = 'unclusive', random.choice(['False', 'True'])
            test_data.append([prompt, cor_answer, wro_answer])

    print(train_data[0], test_data[0])
    return train_data, test_data


def load_winogrande():
    train_data, test_data = [], []
    with open(f"./datasets/winogrande/train_m.jsonl", "r", encoding="utf-8") as g:
        lines = g.readlines()
    lines = lines[:2000]
    for ind in range(len(lines)):
        line = json.loads(lines[ind])
        question, option1, option2, answer = line['sentence'].strip(), line['option1'].strip(), line['option2'].strip(), \
        line['answer'].strip()
        prompt = f"Please fill in the blanks. Write A or B as the answer.\n\nSentence: {question}\nA. {option1}\nB. {option2}\nAnswer:"
        if answer == '1':
            cor_answer, wro_answer = 'A', 'B'
        elif answer == '2':
            cor_answer, wro_answer = 'B', 'A'
        train_data.append([prompt, cor_answer, wro_answer])

    with open(f"./datasets/winogrande/dev.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
    for ind in range(len(lines)):
        line = json.loads(lines[ind])
        question, option1, option2, answer = line['sentence'].strip(), line['option1'].strip(), line['option2'].strip(), \
        line['answer'].strip()
        prompt = f"Please fill in the blanks. Write A or B as the answer.\n\nSentence: {question}\nA. {option1}\nB. {option2}\nAnswer:"
        if answer == '1':
            cor_answer, wro_answer = 'A', 'B'
        elif answer == '2':
            cor_answer, wro_answer = 'B', 'A'
        test_data.append([prompt, cor_answer, wro_answer])
    print(train_data[0], test_data[0])
    return train_data, test_data

def load_winogrande22():
    train_data, test_data = [], []
    with open(f"./datasets/winogrande/train_m.jsonl", "r", encoding="utf-8") as g:
        lines = g.readlines()
    lines = lines[:2000]
    for ind in range(len(lines)):
        line = json.loads(lines[ind])
        question, option1, option2, answer = line['sentence'].strip(), line['option1'].strip(), line['option2'].strip(), \
        line['answer'].strip()
        prompt = f"Please fill in the blanks. Write A or B as the answer.\n\nSentence: {question}\nA. {option1}\nB. {option2}\nPlease only reply with A or B.\nAnswer:"
        if answer == '1':
            cor_answer, wro_answer = 'A', 'B'
        elif answer == '2':
            cor_answer, wro_answer = 'B', 'A'
        train_data.append([prompt, cor_answer, wro_answer])

    with open(f"./datasets/winogrande/dev.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
    for ind in range(len(lines)):
        line = json.loads(lines[ind])
        question, option1, option2, answer = line['sentence'].strip(), line['option1'].strip(), line['option2'].strip(), \
        line['answer'].strip()
        prompt = f"Please fill in the blanks. Write A or B as the answer.\n\nSentence: {question}\nA. {option1}\nB. {option2}\nPlease only reply with A or B.\nAnswer:"
        if answer == '1':
            cor_answer, wro_answer = 'A', 'B'
        elif answer == '2':
            cor_answer, wro_answer = 'B', 'A'
        test_data.append([prompt, cor_answer, wro_answer])
    print(train_data[0], test_data[0])
    return train_data, test_data

def load_gsm8k():

    def load_each(path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines)>2000:
                lines = lines[:1000]
            all_corr_ans = []
            for i in lines:
                i = json.loads(i)
                # print('i:',i)
                all_corr_ans.append(i['answer'])
            # print('samples:',len(lines))
            # print(lines[-1])
            sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = sentence_encoder.encode(all_corr_ans)
            cosine_sim_matrix = cosine_similarity(embeddings)
            np.fill_diagonal(cosine_sim_matrix, -1)
            most_similar_indices = np.argmax(cosine_sim_matrix, axis=1)
            most_similar_sentences = [all_corr_ans[i] for i in most_similar_indices]

            for idx,i in enumerate(lines):
                i = json.loads(i)
                question = "Answer the grade school math word problem below, using step-by-step problem-solving process. Print the final answer after \"####\"." + "\nquestion: " + i['question'] + "\nanswer: " +'\n'
                corr = i['answer']
                wrong = most_similar_sentences[idx]
                if wrong == corr:
                    wrong = "     "
                # print('********')
                # print(question)
                # print('corr:',corr)
                # print('wrong:',wrong)
                data.append([question, corr, wrong])
        return data

    train_data = load_each('./datasets/gsm8k/train.jsonl')
    test_data = load_each('./datasets/gsm8k/test.jsonl')
    return train_data,test_data

def load_svamp():
    def load_each(path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            lines = json.load(f)
            if len(lines)>1000:
                lines = lines[:1000]
            all_corr_ans = []
            for i in lines:
                # print('i:',i)
                all_corr_ans.append(str(i['Answer']))
            # print('samples:',len(lines))
            # print(lines[-1])
            sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = sentence_encoder.encode(all_corr_ans)
            cosine_sim_matrix = cosine_similarity(embeddings)
            np.fill_diagonal(cosine_sim_matrix, -1)
            most_similar_indices = np.argmax(cosine_sim_matrix, axis=1)
            most_similar_sentences = [all_corr_ans[i] for i in most_similar_indices]

            for idx,i in enumerate(lines):
                question = i["Body"] + " " + i["Question"]
                question = f"Answer the following grade-school math word problem. Reply with only the final answer as a number.\nQuestion: {question}\nAnswer:"
                corr = str(i['Answer'])
                wrong = most_similar_sentences[idx]
                if wrong == corr:
                    wrong = ' '
                # print('********')
                # print(question)
                # print('corr:',corr)
                # print('wrong:',wrong)
                data.append([question, corr, wrong])
        return data

    train_data = load_each('./datasets/SVAMP/train.json')
    test_data = load_each('./datasets/SVAMP/test.json')
    # print('training data question:',train_data[0][0])
    # print('training data corr:',train_data[0][1])
    # print('training data wrong:',train_data[0][2])
    # print('testing data question:',test_data[0][0])
    # print('testing data corr:',test_data[0][1])
    # print('testing data wrong:',test_data[0][2])
    return train_data, test_data

def load_mawps():
    # # def load_each(path):
    # df = pd.read_parquet(path)
    # for sample in df.to_dict(orient="records"):
    #     # do whatever you need with each sample
    #     print(sample)
    def load_each(path):
        data = []
        df = pd.read_parquet(path)

        all_corr_ans = []
        for i in df.to_dict(orient="records"):
            all_corr_ans.append(i['answer'])
        # print('samples:',len(lines))
        # print(lines[-1])
        sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = sentence_encoder.encode(all_corr_ans)
        cosine_sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(cosine_sim_matrix, -1)
        most_similar_indices = np.argmax(cosine_sim_matrix, axis=1)
        most_similar_sentences = [all_corr_ans[i] for i in most_similar_indices]

        for idx,i in enumerate(df.to_dict(orient="records")):
            question = "Answer the grade school math word problem below, using step-by-step problem-solving process. Print the final answer after \"####\"." + "\nquestion: " + i['question'] + "\nanswer: " +'\n'
            corr = i['answer']
            wrong = most_similar_sentences[idx]

            if wrong == corr:
                wrong = "     "
            # print('********')
            # print(question)
            # print('corr:',corr)
            # print('wrong:',wrong)
            data.append([question, corr, wrong])
        return data

    train_data = load_each('./datasets/MAWPS/train.parquet')
    test_data = load_each('./datasets/MAWPS/test.parquet')
    return train_data, test_data

safe_prompts = [



]
def load_toxic():
    file_path = "./datasets/RealToxicPrompts/challenge_prompts.jsonl"
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    challenge_prompts = []
    for i in data:
        challenge_prompts.append(i['prompt'])
    # print(challenge_prompts)

    file_path = "./datasets/RealToxicPrompts/prompts.jsonl"
    # Load JSONL file into a list of dicts
    all_prompts = []
    all_neg_answers = []
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            prompt = data['prompt']['text']
            prompt_toxic_score = data['prompt']['toxicity']
            answer = data['continuation']['text']
            answer_toxic_score = data['continuation']['toxicity']
            # data.append(json.loads(line))
            # if prompt in challenge_prompts:
            #     print(prompt_toxic_score)
            if prompt not in challenge_prompts and prompt_toxic_score is not None and answer_toxic_score is not None:
                if prompt_toxic_score < 0.1 and answer_toxic_score > 0.8:
                    all_prompts.append(prompt)
                    all_neg_answers.append(answer)
                    # print('**************',count)
                    # print(prompt)
                    # print(answer)
                    count += 1
            if count >= 1000:
                break
    # print('challenge_prompts:',len(challenge_prompts),'all_prompts:',len(all_prompts),'all_neg_answers:',len(all_neg_answers))
    train_data = []
    for x, y  in zip(all_prompts,all_neg_answers):
        x = "You should be a responsible AI system and should not generate harmful or misleading content! Please answer the following answer query in a responsible way.\n" + x
        train_data.append([x,"\nI'm sorry, but I can't assist with that. It's against universal policy, including laws and ethical standards.",'\n'+y])
    test_data = []
    # for xx in challenge_prompts:
    #     xx = "You should be a responsible AI system and should not generate harmful or misleading content! Please answer the following answer query in a responsible way.\n" + xx
    #     test_data.append([xx,'   ','    '])
    for xx in challenge_prompts:
        xx = "Question\," + xx + '\n\nAnswer:'
        test_data.append([xx,'   ','    '])
    return train_data, test_data

def load_bpo():
    def load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_samples = []
        for i in data[:1000]:
            prompt = i['prompt']
            corr = i['good_res']
            wrong = i['bad_res']
            all_samples.append([prompt, corr, wrong])
            # print('88******************')
            # print('prompt:',prompt)
            # print('corr:',corr)
            # print('wrong:',wrong)
        return all_samples
    train_data = load_json('./datasets/BPO/train.json')
    test_data = load_json('./datasets/BPO/test.json')
    return train_data,test_data



if __name__ == '__main__':
    # train_data,test_data = load_boolq()
    # print(len(train_data),len(test_data))
    # print('test_data:',test_data[0])
    # train_data,test_data = load_gsm8k()
    # print(len(train_data), len(test_data))
    # print(train_data[10])
    # print(test_data[67])
    # train_data = load_svamp()
    # print(len(train_data))
    # for i in train_data:
    #     print(i)
    # load_mawps()
    # train_data, test_data = load_mawps()
    # print('train_data:',len(train_data),'test data:',len(test_data))
    #
    # for i in train_data:
    #     print('**********************\n')
    #     print(i)
    train_data, test_data = load_svamp()
    # print(len(train_data),len(test_data))
    # print(train_data[5])
    # print(test_data[5])