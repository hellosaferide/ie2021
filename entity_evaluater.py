from tqdm import tqdm
from transformers import BertTokenizer
import torch

model_path = 'pretrained_model/RoBERTa_zh'
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower=True)
maxlen = 512
no_cuda = False
device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
import numpy as np
import unicodedata
import json


def is_string(s):
    """判断是否是字符串
    """
    return isinstance(s, str)


def _is_control(ch):
    """控制类字符判断
    """
    return unicodedata.category(ch) in ('Cc', 'Cf')


def _is_special(ch):
    """
    判断是不是有特殊含义的符号
    """
    return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')


def stem(token):
    """获取token的“词干”（如果是##开头，则自动去掉##）
    """
    if token[:2] == '##':
        return token[2:]
    else:
        return token


def rematch(text, tokens, do_lower_case=True):
    """给出原始的text和tokenize后的tokens的映射关系
    """

    normalized_text, char_mapping = '', []
    for i, ch in enumerate(text):
        if do_lower_case:
            ch = unicodedata.normalize('NFD', ch)
            ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
        ch = ''.join([
            c for c in ch
            if not (ord(c) == 0 or ord(c) == 0xfffd or _is_control(c))
        ])
        normalized_text += ch
        char_mapping.extend([i] * len(ch))
    text = normalized_text.lower()

    token_mapping, offset = [], 0
    for token in tokens:
        #
        if _is_special(token):
            token_mapping.append([])
        else:
            token = stem(token)
            start = text[offset:].index(token) + offset
            end = start + len(token)
            token_mapping.append(char_mapping[start:end])
            offset = end

    return token_mapping


def change(tokens):
    stem_token = []
    for token in tokens:
        token = stem(token)
        stem_token.append(token)
    return stem_token


def extract_entity_pair(text, model):
    tokens = tokenizer.tokenize(text)
    # print(tokens)
    token = tokenizer.encode_plus(text, max_length=maxlen, truncation=True)
    stem_tokens = change(tokens)
    token_ids, segment_ids = token['input_ids'], token['token_type_ids']
    sub_token_ids = torch.tensor([token_ids]).to(device=device)
    sub_segment_ids = torch.tensor([segment_ids]).to(device=device)

    # TODO rematching
    # mapping = rematch(text, tokens)
    subject_pred = model(input_ids=sub_token_ids, segment_ids=sub_segment_ids,
                         batch_size=1, sub_train=True, device=device)
    subject_pred = subject_pred.view(1, -1, 2)
    subject_pred = subject_pred.detach().cpu().numpy()
    # print("subject_pred:", subject_pred)

    start = np.where(subject_pred[0, :, 0] > 0.5)[0]
    end = np.where(subject_pred[0, :, 1] > 0.5)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))
    if subjects:
        so = []
        obj_token_ids = np.repeat([token_ids], len(subjects), 0)
        obj_segment_ids = np.repeat([segment_ids], len(subjects), 0)
        obj_token_ids = torch.tensor(obj_token_ids).long().to(device)
        obj_segment_ids = torch.tensor(obj_segment_ids).long().to(device)
        sub_len = len(subjects)
        subjects = torch.tensor(subjects).to(device)
        # print(subjects)
        subject_pred, object_preds = model(input_ids=obj_token_ids, segment_ids=obj_segment_ids, subject_ids=subjects,
                                           batch_size=sub_len, sub_train=True, obj_train=True, device=device)
        so_index = []
        for subject, object_pred in zip(subjects, object_preds):
            # print("object_pred:", object_pred)
            # print("subject:", subject)
            object_pred = object_pred.detach().cpu().numpy()
            start = np.where(object_pred[:, 0] > 0.5)[0]
            end = np.where(object_pred[:, 1] > 0.5)[0]
            objects = []
            for i in start:
                j = end[end >= i]
                if len(j) > 0:
                    j = j[0]
                    objects.append((i, j))
            for obj in objects:
                # (subject, obj)
                # print("object:", obj)
                so_index.append((subject, obj))
                sub_text_list = stem_tokens[subject[0] - 1: subject[1]]
                obj_text_list = stem_tokens[obj[0] - 1: obj[1]]
                sub_text = ""
                for sub in sub_text_list:
                    if _is_special(sub):
                        continue
                    sub_text += str(sub)

                obj_text = ""
                for t in obj_text_list:
                    if _is_special(t):
                        continue
                    obj_text += t
                if subject == obj:
                    continue
                so.append((sub_text, obj_text))
        return so
    else:
        return []


def evaluate(dev_data, model):
    # 评估函数，计算f1、precision、recall
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('dev_pred.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for data in dev_data:
        # print(data)
        R = extract_entity_pair(data['text'], model)
        T = []
        for spo in data['spo_list']:
            # print(type(spo))
            # print(spo)
            # print("spo:", spo)
            s = spo[0]
            o = spo[2]
            T.append((s, o))
        # print("R:", R)
        # print("T:", T)
        T = set(T)
        R = set(R)

        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        s = json.dumps({
            'text': data['text'],
            'so_list': list(T),
            'so_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        }, ensure_ascii=False, indent=4)
        f.write(s + '\n')
    pbar.close()
    return f1, precision, recall
