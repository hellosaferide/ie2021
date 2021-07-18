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
from load import load_schema
from util import combine_spoes, search, SPO, sequence_padding


def extract_relation(text, realtion_model, so, subject_object):
    tokens = tokenizer.tokenize(text)
    id2predicate, _, _ = load_schema()
    # print(tokens)
    token = tokenizer.encode_plus(text, max_length=maxlen, truncation=True)
    token_ids, segment_ids = token['input_ids'], token['token_type_ids']
    sub_token_ids = np.repeat([token_ids], len(so), 0)
    sub_segment_ids = np.repeat([segment_ids], len(so), 0)

    batch_token_ids, batch_segment_ids, batch_attention_mask = [], [], []
    for i in range(len(so)):
        subject = so[i][0]
        objects = so[i][1]
        token_ids = sub_token_ids[i][1:]
        segment_ids = sub_segment_ids[i]
        cls = np.array([102])
        sep = np.array([103])
        sub_sep = np.append(subject, sep)
        obj_sep = np.append(objects, sep)
        obj_sub_sep = np.append(sub_sep, obj_sep)
        token_ids = np.append(obj_sub_sep, token_ids)
        token_ids = np.append(cls, token_ids)
        segment_ids = np.append(np.zeros((1, len(subject) + len(objects) + 2)), segment_ids)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)

    max_length = 0
    batch_token_id = []
    for token_id in batch_token_ids:
        if len(token_id) > max_length:
            max_length = len(token_id)
    for token_id in batch_token_ids:
        batch_token_id.append(sequence_padding(token_id, max_length))
    _batch_token_ids = np.array(batch_token_id)

    max_length = 0
    batch_segment_id = []
    for segment_id in batch_segment_ids:
        if len(segment_id) > max_length:
            max_length = len(segment_id)
    for segment_id in batch_segment_ids:
        batch_segment_id.append(sequence_padding(segment_id, max_length))
    _batch_segment_ids = np.array(batch_segment_id)
    batch_token_ids = torch.tensor(_batch_token_ids).long().to(device)
    batch_segment_ids = torch.tensor(_batch_segment_ids).long().to(device)

    predicate_output = realtion_model(input_ids=batch_token_ids,
                                      attention_mask=None,
                                      token_type_ids=batch_segment_ids,
                                      labels=None, )
    logits = predicate_output[0]


    text_index = 0
    spoes = []
    for output in logits:
        index = torch.argmax(output, 0).item()
        predicate = id2predicate[index]
        subject = subject_object[text_index][0]
        objects = subject_object[text_index][1]
        spoes.append((subject, predicate, objects))
        text_index += 1
    return spoes


def evaluate(dev_data, model):
    # 评估函数，计算f1、precision、recall
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('dev_pred.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for data in dev_data:
        token = tokenizer.encode_plus(
            data['text'], max_length=maxlen, truncation=True
        )
        so = []
        subject_object = []
        for s, p, o in data['spo_list']:
            subject_object.append((s, o))
            s = tokenizer.encode_plus(s)['input_ids'][1: -1]
            o = tokenizer.encode_plus(o)['input_ids'][1: -1]
            so.append((s, o))
        # print(data)
        # R = combine_spoes(extract_relation(data['text'], model, so, subject_object))
        # T = combine_spoes(data['spo_list'])

        R = extract_relation(data['text'], model, so, subject_object)
        T = data['spo_list']

        # R = set([SPO(spo) for spo in R])
        # T = set([SPO(spo) for spo in T])
        R = set(R)
        T = set(T)
        X += len(R & T)
        Y += len(T)
        acc = X / Y
        pbar.update()
        pbar.set_description(
            'acc: %.5f' % (acc)
        )
        print(R)
        print(T)
        s = json.dumps({
            'text': data['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        }, ensure_ascii=False, indent=4)
        f.write(s + '\n')
    pbar.close()
    return acc
