import numpy as np
# from bert4keras.tokenizers import Tokenizer
from transformers import BertTokenizer
import os
import torch.utils.data as Data
from load import load_schema
import torch
from tqdm import tqdm
import json

model_path = 'pretrained_model/RoBERTa_zh'
# tokenizer_k = Tokenizer(os.path.join(model_path, 'vocab.txt'), do_lower_case=True)
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower=True)
maxlen = 512
no_cuda = False
device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
from random import choice
import unicodedata


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


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def sequence_padding(input, length, padding=0):
    """
    Numpy函数，将序列padding到同一长度
    """

    if len(input) < length:
        output = np.concatenate((input, [padding] * (length - len(input))))
    else:
        output = input[:length]

    return output


def combine_spoes(spoes):
    new_spoes = {}
    for s, p, o in spoes:
        print(spoes)
        print(s)
        print(p)
        print(o)
        p1, p2 = p.split('_')
        if (s, p1) in new_spoes:
            new_spoes[(s, p1)][p2] = o
        else:
            new_spoes[(s, p1)] = {p2: o}

    return [(k[0], k[1], v) for k, v in new_spoes.items()]


class SPO(tuple):

    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(
                sorted([
                    (k, tuple(tokenizer.tokenize(v))) for k, v in spo[2].items()
                ])
            )
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


def extract_spoes(text, model):
    """
    抽取输入text所包含的三元组
    """
    id2predicate, predicate2id, n = load_schema()
    tokens = tokenizer.tokenize(text)
    token = tokenizer.encode_plus(text, max_length=maxlen, truncation=True)


    sub_token_ids = torch.tensor([token_ids]).to(device=device)
    sub_segment_ids = torch.tensor([segment_ids]).to(device=device)
    # TODO mapping
    mapping = rematch(text, tokens)

    subject_pred = model(input_ids=sub_token_ids, segment_ids=sub_segment_ids,
                         batch_size=1, sub_train=True, device=device)

    # 抽取subject
    subject_pred = subject_pred.view(1, -1, 2)
    subject_pred = subject_pred.detach().cpu().numpy()
    # print("subject_pred:", subject_pred)

    start = np.where(subject_pred[0, :, 0] > 0.4)[0]
    end = np.where(subject_pred[0, :, 1] > 0.4)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))

    if subjects:
        spoes = []
        obj_token_ids = np.repeat([token_ids], len(subjects), 0)
        obj_segment_ids = np.repeat([segment_ids], len(subjects), 0)
        obj_token_ids = torch.tensor(obj_token_ids).to(device)
        obj_segment_ids = torch.tensor(obj_segment_ids).to(device)
        sub_len = len(subjects)
        subjects = torch.tensor(subjects).to(device)
        # 传入subject， object
        subject_pred, object_preds = model(input_ids=obj_token_ids, segment_ids=obj_segment_ids, subject_ids=subjects,
                                           batch_size=len(subjects), sub_train=True, obj_train=True, device=device)


        for subject, object_pred in zip(subjects, object_preds):
            # TODO 估计存在问题
            print("object_pred:", object_pred)
            object_pred = object_pred.detach().cpu().numpy()
            start = np.where(object_pred[:, 0] > 0.5)[0]
            end = np.where(object_pred[:,   1] > 0.5)[0]
            # print(start.size)
            # print(end.size)
            objects = []
            for i in start:
                j = end[end >= i]
                if len(j) > 0:
                    j = j[0]
                    objects.append((i, j))
            print("object:", objects)
            for obj in objects:
                # (subject, obj)作为输入
                _, _, predicate = model(input_ids=sub_token_ids, segment_ids=sub_segment_ids, subject_ids=subject,
                                        object_ids=obj, batch_size=1, sub_train=True, obj_train=True,
                                        predicate_train=True, device=device)
                print("predicate:", predicate)
                predicate = predicate[0]
                print("predicate:", predicate)
                # 找出值最大的索引
                predicate_index = torch.argmax(predicate, dim=1)
                # TODO
                print("predicate_index:", predicate_index)
                if len(mapping[subject[0]]) > 0 and len(mapping[subject[1]]) > 0 and len(mapping[obj[0]]) > 0 and len(
                        mapping[obj[1]]) > 0:
                    spoes.append((mapping[subject[0]][0], mapping[subject[1]][-1]), predicate_index,
                                 (mapping[obj[0]][0], mapping[obj[1]][-1]))
        # print([(text[s[0]:s[1] + 1], id2predicate[p], text[o[0]:o[1] + 1]) for s, p, o, in spoes])
        return [(text[s[0]:s[1] + 1], id2predicate[p], text[o[0]:o[1] + 1]) for s, p, o, in spoes]
    else:
        return []


def evaluate(dev_data, model):
    # 评估函数，计算f1、precision、recall
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('dev_pred.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for data in dev_data:
        R = combine_spoes(extract_spoes(data['text'], model))
        T = combine_spoes(data['spo_list'])
        R = set([SPO(spo) for spo in R])
        T = set([SPO(spo) for spo in T])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X * Y / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        s = json.dumps({
            'text': data['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },
            ensure_ascii=False,
            indent=4)
        f.write(s + '\n')
    pbar.close()
    return f1, precision, recall


class data_generator:
    """
    数据生成器
    """

    def __init__(self, data, batch_size=20, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None

        self.buffer_size = buffer_size

    def __len__(self):
        return self.steps

    def get_data(self):
        batch_token_ids, batch_segment_ids, batch_attention_mask = [], [], []
        batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
        batch_obj_ids, batch_relation_label = [], []
        indices = list(range(len(self.data)))
        np.random.shuffle(indices)
        _, predicate2id, _ = load_schema()
        for i in indices:
            d = self.data[i]
            token = tokenizer.encode_plus(
                d['text'], max_length=maxlen, truncation=True
            )
            token_ids, segment_ids, attention_mask = token['input_ids'], token['token_type_ids'], token[
                'attention_mask']

            spoes = {}
            for s, p, o in d['spo_list']:
                s = tokenizer.encode_plus(s)['input_ids'][1: -1]
                p = predicate2id[p]
                o = tokenizer.encode_plus(o)['input_ids'][1: -1]
                s_idx = search(s, token_ids)
                o_idx = search(o, token_ids)
                if s_idx != -1 and o_idx != -1 and len(s) > 0 and len(o) > 0:
                    s = (s_idx, s_idx + len(s) - 1)
                    o = (o_idx, o_idx + len(o) - 1, p)

                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)
            if spoes:
                # subject标签
                subject_labels = np.zeros((len(token_ids), 2))
                for s, _ in spoes.items():
                    subject_labels[s[0], 0] = 1
                    subject_labels[s[1], 1] = 1

                sub_key = list(spoes.keys())
                index = range(len(sub_key))
                start_index = choice(index)
                sub_start = sub_key[start_index][0]
                sub_end = sub_key[start_index][1]
                subject_ids = (sub_start, sub_end)
                object = spoes[subject_ids]
                index = range(len(object))

                start_index = choice(index)
                obj_start = object[start_index][0]
                obj_end = object[start_index][1]
                object_ids = (obj_start, obj_end)

                predicate = object[start_index][2]
                # predicate_label = np.zeros((len(predicate2id), 1))
                # predicate_label[predicate][0] = 1
                object_labels = np.zeros((len(token_ids), 2))
                for s, o in spoes.items():
                    for obj in o:
                        object_labels[obj[0], 0] = 1
                        object_labels[obj[1], 1] = 1

                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_subject_labels.append(subject_labels)
                batch_attention_mask.append(attention_mask)
                batch_object_labels.append(object_labels)
                batch_obj_ids.append(object_ids)
                batch_subject_ids.append(subject_ids)
                batch_relation_label.append(predicate)

        max_length = 0
        batch_token_id = []
        for token_id in batch_token_ids:
            if len(token_id) > max_length:
                max_length = len(token_id)
        for token_id in batch_token_ids:
            batch_token_id.append(sequence_padding(token_id, max_length))
        batch_token_ids = np.array(batch_token_id)

        max_length = 0
        batch_segment_id = []
        for segment_id in batch_segment_ids:
            if len(segment_id) > max_length:
                max_length = len(segment_id)
        for segment_id in batch_segment_ids:
            batch_segment_id.append(sequence_padding(segment_id, max_length))
        batch_segment_ids = np.array(batch_segment_id)

        max_length = 0
        batch_subject_label = []
        for subject_label in batch_subject_labels:
            if len(subject_label) > max_length:
                max_length = len(subject_label)
        for subject_label in batch_subject_labels:
            batch_subject_label.append(sequence_padding(
                subject_label,
                max_length,
                padding=np.zeros(2)
            ))
        batch_subject_labels = np.array(batch_subject_label)

        max_length = 0
        batch_obj_label = []
        for object_label in batch_object_labels:
            if len(object_label) > max_length:
                max_length = len(object_label)
        for object_label in batch_object_labels:
            batch_obj_label.append(sequence_padding(
                object_label,
                max_length,
                padding=np.zeros(2)
            ))
        batch_obj_label = np.array(batch_obj_label)
        batch_subject_ids = np.array(batch_subject_ids)
        batch_obj_ids = np.array(batch_obj_ids)

        max_length = 0
        attention_mask_list = []
        for attention_mask in batch_attention_mask:
            if len(attention_mask) > max_length:
                max_length = len(attention_mask)
        for attention_mask in batch_attention_mask:
            attention_mask_list.append(sequence_padding(attention_mask, max_length))
        batch_attention_mask = np.array(attention_mask_list)

        batch_relation_label = np.array(batch_relation_label)
        return [batch_token_ids, batch_segment_ids, batch_subject_labels,
                batch_obj_label, batch_subject_ids, batch_obj_ids, batch_attention_mask, batch_relation_label]


class Dataset(Data.Dataset):

    def __init__(self, _batch_token_ids, _batch_segment_ids, _batch_subject_labels, _batch_subject_ids,
                 _batch_object_labels, _batch_obj_ids, _batch_attention_mask, _batch_relation_label):
        self.batch_token_ids = _batch_token_ids
        self.batch_segment_ids = _batch_segment_ids
        self.batch_subject_labels = _batch_subject_labels
        self.batch_subject_ids = _batch_subject_ids
        self.batch_object_labels = _batch_object_labels
        self.batch_obj_ids = _batch_obj_ids
        self.batch_attention_mask = _batch_attention_mask
        self.batch_relation_label = _batch_relation_label
        self.len = len(self.batch_token_ids)

    def __getitem__(self, index):
        return self.batch_token_ids[index], self.batch_segment_ids[index], self.batch_subject_labels[index] \
            , self.batch_subject_ids[index], self.batch_object_labels[index], self.batch_obj_ids[index], \
               self.batch_attention_mask[index], self.batch_relation_label[index]

    def __len__(self):
        return self.len


def collate_fn(data):
    batch_token_ids = np.array([item[0] for item in data], np.int32)
    batch_segment_ids = np.array([item[1] for item in data], np.int32)
    batch_subject_labels = np.array([item[2] for item in data], np.int32)
    batch_subject_ids = np.array([item[3] for item in data], np.int32)
    batch_object_labels = np.array([item[4] for item in data], np.int32)
    batch_obj_ids = np.array([item[5] for item in data], np.int32)
    batch_attention_mask = np.array([item[6] for item in data], np.int32)
    batch_relation_label = np.array([item[7] for item in data], np.int32)

    return {
        'batch_token_ids': torch.LongTensor(batch_token_ids),  # targets_i
        'batch_segment_ids': torch.LongTensor(batch_segment_ids),
        'batch_subject_labels': torch.FloatTensor(batch_subject_labels),
        'batch_subject_ids': torch.LongTensor(batch_subject_ids),
        'batch_object_labels': torch.FloatTensor(batch_object_labels),
        'batch_obj_ids': torch.LongTensor(batch_obj_ids),
        'batch_attention_mask': torch.LongTensor(batch_attention_mask),
        'batch_relation_label': torch.FloatTensor(batch_relation_label)
    }
