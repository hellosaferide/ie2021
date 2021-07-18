from datetime import datetime

import torch.nn as nn
import torch.utils.data as Data
import json
import numpy as np
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
from util import Dataset, collate_fn
from realtion import relation_model
from util import data_generator, sequence_padding
from load import load_data, load_schema
import torch
import os
import config
from logger import Logger
from relation_evaluator import evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_path = 'pretrained_model/RoBERTa_zh'
config = config.relation_config
hyper_parameters = config['hyper_parameters']
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
max_len = hyper_parameters["max_len"]
weight_decay = hyper_parameters["weight_decay"]
learning_rate = hyper_parameters["learning_rate"]
adam_epsilon = hyper_parameters["adam_epsilon"]
epochs = hyper_parameters["epochs"]
rewarm_epoch_num = hyper_parameters["rewarm_epoch_num"]
T_mult = hyper_parameters["T_mult"]
output_dir = hyper_parameters["output_dir"]
train_path = hyper_parameters["train_path"]
valid_path = hyper_parameters["valid_path"]
model_path = hyper_parameters["model_path"]
batch_size = hyper_parameters["batch_size"]
task_learning_rate = hyper_parameters["task_learning_rate"]
warmup_proportion = hyper_parameters["warmup_proportion"]
no_cuda = False
device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
# device = "cpu"
best_acc = 0.0

if __name__ == "__main__":

    print("load data")
    train_data = load_data(train_path)
    valid_data = load_data(valid_path)

    print("generate data")
    dg = data_generator(train_data)
    de_dev = data_generator(valid_data)
    print("get data")

    _batch_token_ids, _batch_segment_ids, _batch_subject_labels, _batch_obejct_labels, _batch_subject_ids, _batch_object_ids, _batch_attention_mask, _batch_relation_label = dg.get_data()

    # _batch_token_ids, _batch_segment_ids, _batch_attention_mask, _batch_subject_ids, _batch_object_ids
    batch_token_ids, batch_segment_ids,  batch_attention_mask = [], [], []

    for i in range(len(_batch_token_ids)):
        token_ids = _batch_token_ids[i]
        segment_ids = _batch_segment_ids[i]
        attention_mask = _batch_attention_mask[i]
        subject = token_ids[_batch_subject_ids[i][0]: _batch_subject_ids[i][1] + 1]
        objects = token_ids[_batch_object_ids[i][0]: _batch_object_ids[i][1] + 1]
        token_ids = _batch_token_ids[i][1:]
        cls = np.array([102])
        sep = np.array([103])
        sub_sep = np.append(subject, sep)
        obj_sep = np.append(objects, sep)
        obj_sub_sep = np.append(sub_sep, obj_sep)
        token_ids = np.append(obj_sub_sep, token_ids)
        token_ids = np.append(cls, token_ids)
        segment_ids = np.append(np.zeros((1, len(subject) + len(objects) + 2)), segment_ids)
        attention_mask = np.append(np.ones((1, len(subject) + len(objects) + 2)), attention_mask)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_attention_mask.append(attention_mask)

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

    max_length = 0
    attention_mask_list = []
    for attention_mask in batch_attention_mask:
        if len(attention_mask) > max_length:
            max_length = len(attention_mask)
    for attention_mask in batch_attention_mask:
        attention_mask_list.append(sequence_padding(attention_mask, max_length))
    _batch_attention_mask = np.array(attention_mask_list)

    train_dataset = Dataset(_batch_token_ids, _batch_segment_ids, _batch_subject_labels, _batch_subject_ids,
                            _batch_obejct_labels, _batch_object_ids, _batch_attention_mask, _batch_relation_label)


    train_dataloader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # print("index")
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    log_dir = 'train/' + TIMESTAMP
    print(log_dir)
    logger = Logger(log_dir)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=55)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay) and 'bert' not in n)],
         'weight_decay': weight_decay, 'lr': task_learning_rate},  # n wei 层的名称, p为参数

        {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay) and 'bert' in n)],
         'weight_decay': weight_decay},  # n wei 层的名称, p为参数

        {'params': [p for n, p in param_optimizer if (any(nd in n for nd in no_decay) and 'bert' not in n)],
         'weight_decay': 0.0, 'lr': task_learning_rate},
        # 如果是 no_decay 中的元素则衰减为 0

        {'params': [p for n, p in param_optimizer if (any(nd in n for nd in no_decay) and 'bert' in n)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    t_total = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total * warmup_proportion), t_total)
    model.to(device)
    model.train()
    print("start training")
    f = open(r'relation.txt', 'a', encoding='utf8')
    f.write(log_dir)
    for epoch in range(0, epochs):
        total_sub_loss, total_obj_loss, total_predicate_loss = 0.0, 0.0, 0.0
        total_loss = 0.0
        print("epoch:", epoch)
        step = 0
        for batch_data in tqdm(train_dataloader):
            step += 1
            batch_token_ids = batch_data['batch_token_ids'].to(device)
            batch_segment_ids = batch_data['batch_segment_ids']
            # batch_subject_ids = batch_data['batch_subject_ids'].to(device)
            # batch_obj_ids = batch_data['batch_obj_ids'].to(device)
            batch_attention_mask = batch_data['batch_attention_mask']
            batch_relation_label = batch_data['batch_relation_label'].long().to(device)
            batch_segment_ids = batch_segment_ids.long().to(device)
            batch_attention_mask = batch_attention_mask.long().to(device)
            loss = model(batch_token_ids, batch_attention_mask, batch_segment_ids, labels=batch_relation_label)[0]
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1)
            total_loss += loss.item()
            total_loss = round(total_loss, 4)
            optimizer.step()
            scheduler.step()
            print("loss:{}".format(loss.item()))

            if (step + 1) % 100 == 0:
                info = {'loss': loss.item(), 'learning_rate': optimizer.param_groups[0]['lr'],
                        'task_learning_rate': task_learning_rate}
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step + 1)

        avg_loss = total_loss / len(train_dataloader)
        print("avg_loss:", avg_loss)
        f.write("avg_loss:" + str(avg_loss) + '\n')
        model.eval()
        acc = evaluate(valid_data, model)
        print(acc)

        if acc > best_acc:
            print("Best acc", acc)
            print("Saving Model......")
            best_acc = acc
            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(model.state_dict(), output_model_file)  # 仅保存学习到的参数
            f.write(str(epoch) + '\t' + str(acc) + '\n')
        else:
            f.write(str(epoch) + '\t' + str(acc) + '\n')