from datetime import datetime

import torch.nn as nn
import torch.utils.data as Data
import json
import numpy as np
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from util import Dataset, collate_fn
from REmodel import relation_model
from util import data_generator
from load import load_data, load_schema
import torch
import os
import random
import config
from logger import Logger
from entity_evaluater import evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_path = 'pretrained_model/RoBERTa_zh'
config = config.train_config
hyper_parameters = config['hyper_parameters']
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
    train_dataset = Dataset(_batch_token_ids, _batch_segment_ids, _batch_subject_labels, _batch_subject_ids,
                            _batch_obejct_labels, _batch_object_ids, _batch_attention_mask, _batch_relation_label)
    """
    for index in range(len(_batch_token_ids)):
        print(_batch_subject_ids[index])
        print("_batch_object_ids:", _batch_object_ids[index])
        print("_batch_obejct_label s:", _batch_obejct_labels[index].shape)
    """

    train_dataloader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # print("index")
    """
        for step, batch_data in enumerate(train_dataloader):

        print(batch_data['batch_token_ids'])
        print(batch_data['batch_subject_labels'])
        print(batch_data['batch_subject_ids'])
        print(batch_data['batch_obj_ids'])
    """
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    log_dir = 'train/' + TIMESTAMP
    print(log_dir)
    logger = Logger(log_dir)
    model = relation_model.from_pretrained(model_path, num_labels=2, output_hidden_states=True)
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
    f = open(r'loss.txt', 'a', encoding='utf8')
    step = 0
    for epoch in range(0, epochs):
        total_sub_loss, total_obj_loss, total_predicate_loss = 0.0, 0.0, 0.0
        total_loss = 0.0
        print("epoch:", epoch)
        for batch_data in tqdm(train_dataloader):
            step += 1
            batch_token_ids = batch_data['batch_token_ids'].to(device)
            batch_segment_ids = batch_data['batch_segment_ids']
            batch_subject_labels = batch_data['batch_subject_labels'].to(device)
            batch_subject_ids = batch_data['batch_subject_ids'].to(device)
            batch_object_labels = batch_data['batch_object_labels'].to(device)
            batch_obj_ids = batch_data['batch_obj_ids'].to(device)
            batch_attention_mask = batch_data['batch_attention_mask']
            batch_relation_label = batch_data['batch_relation_label'].long().to(device)
            batch_segment_ids = batch_segment_ids.long().to(device)
            batch_attention_mask = batch_attention_mask.long().to(device)
            sub_output, obj_output = model(input_ids=batch_token_ids,
                                           attention_mask=batch_attention_mask,
                                           segment_ids=batch_segment_ids,
                                           subject_labels=batch_subject_labels,
                                           obj_labels=batch_object_labels,
                                           subject_ids=batch_subject_ids,
                                           batch_size=batch_size,
                                           sub_train=True,
                                           obj_train=True,
                                           device=device)

            sub_loss, _ = sub_output[0: 2]
            obj_loss, _ = obj_output[0: 2]
            # predicate_loss, _ = predicate_output[0: 2]
            loss = torch.add(sub_loss, obj_loss)
            # loss = torch.add(loss, predicate_loss)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1)
            total_loss += loss.item()
            total_loss = round(total_loss, 4)
            optimizer.step()
            scheduler.step()
            print("loss:{}, sub_loss:{}, obj_loss:{}".format(loss.item(), sub_loss.item(), obj_loss.item()))


            if (step + 1) % 100 == 0:
                info = {'loss': loss.item(), 'sub_loss': sub_loss.item(), 'obj_loss': obj_loss,
                        'learning_rate': optimizer.param_groups[0]['lr'], 'task_learning_rate': task_learning_rate}
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step + 1)

        # nohup python -u run_ie.py > train.log 2>&1 &
        avg_loss = total_loss / len(train_dataloader)
        # print("avg_loss:", avg_loss)
        f.write("avg_loss:" + str(avg_loss) + '\n')
        model.eval()
        f1, precision, recall = evaluate(valid_data, model)
        print(f1, precision, recall)

        if f1 > best_acc:
            print("Best F1", f1)
            print("Saving Model......")
            best_acc = f1
            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(model.state_dict(), output_model_file)  # 仅保存学习到的参数
            f.write(str(epoch) + '\t' + str(f1) + '\t' + str(precision) + '\t' + str(recall) + '\n')
