from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn
import torch
import math
from torch.nn import functional as F
import numpy as np

from util import sequence_padding


def batch_gather(data: torch.Tensor, index: torch.Tensor):
    # print("data:", data)
    index = index.unsqueeze(-1)
    # print("index:", index)
    index = index.expand(data.size()[0], index.size()[1], data.size()[2])
    # print("index:", index)
    return torch.gather(data, 1, index)


def extrac_subject_1(sequence_output, subject_ids):
    # print("subject_ids[:, :1]:", subject_ids[:, :1])
    # print("subject_ids[:, 1:]:", subject_ids[:, 1:])
    start = batch_gather(sequence_output, subject_ids[:, :1])
    end = batch_gather(sequence_output, subject_ids[:, 1:])
    return start, end


class relation_model(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.obj_label = 110
        self.linear = nn.Linear(768, 768)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, self.num_labels)
        self.sub_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.obj_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.relu = nn.ReLU()
        self.sub_pos_emb = nn.Embedding(512, 768)

    def forward(self, input_ids=None, attention_mask=None, segment_ids=None, subject_labels=None, obj_labels=None,
                subject_ids=None, batch_size=None, sub_train=False, obj_train=False, device=None):

        # shared encoder
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=segment_ids,
        )
        # print(input_ids)
        sequence_output = output[0]  # last_hidden_state
        # TODO Layer Normalization
        sequence_output = self.LayerNorm(sequence_output)
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.relu(self.linear(sequence_output))
        sequence_output = self.dropout(sequence_output)
        if sub_train:
            sub_logits = self.sub_classifier(sequence_output)
            loss_fuc = nn.BCELoss(reduction='none')
            outputs = (sub_logits,)
            sigmoid = nn.Sigmoid()
            active_sub_logits = sub_logits.view(-1, self.num_labels)
            active_sub_logits = sigmoid(active_sub_logits)
            # TODO 为啥要加平方
            active_sub_logits = active_sub_logits ** 2
            if subject_labels is not None:
                active_labels = subject_labels.view(-1, self.num_labels).float()
                loss = loss_fuc(active_sub_logits, active_labels)
                loss = loss.view(-1, input_ids.size()[-1], 2)
                sub_loss = torch.mean(loss, 2)
                sub_loss = torch.sum(attention_mask * sub_loss) / torch.sum(attention_mask)
                sub_outputs = (sub_loss,) + outputs
            else:
                sub_outputs = active_sub_logits.view(-1, input_ids.size()[-1], 2)

        if obj_train == True:
            hidden_states = output[2][-2]
            hidden_states_1 = output[2][-3]
            hidden_states = self.dropout(hidden_states)
            loss_obj = nn.BCELoss(reduction='none')
            loss_sig = nn.Sigmoid()
            sub_pos_start = self.sub_pos_emb(subject_ids[:, :1]).to(device)
            sub_pos_end = self.sub_pos_emb(subject_ids[:, 1:]).to(device)
            subject_start_last, subject_end_last = extrac_subject_1(sequence_output, subject_ids)
            subject_start_1, subject_end_1 = extrac_subject_1(hidden_states_1, subject_ids)
            subject_start, subject_end = extrac_subject_1(hidden_states, subject_ids)
            subject_start = subject_start.to(device)
            subject_end = subject_end.to(device)

            subject = (subject_end_last + subject_start_last + sub_pos_start + sub_pos_end + subject_start_1 + subject_end_1 + subject_start + subject_end).to(device)

            batch_token_ids_obj = torch.add(sequence_output, subject)
            batch_token_ids_obj = self.LayerNorm(batch_token_ids_obj)
            batch_token_ids_obj = self.dropout(batch_token_ids_obj)
            batch_token_ids_obj = self.relu(self.linear(batch_token_ids_obj))
            batch_token_ids_obj = self.dropout(batch_token_ids_obj)
            obj_logits = self.obj_classifier(batch_token_ids_obj)
            active_obj_logits = loss_sig(obj_logits)
            active_obj_logits = active_obj_logits.view(-1, self.num_labels)
            active_obj_logits = active_obj_logits ** 2
            obj_outputs = (obj_logits,)
            if obj_labels is not None:
                active_labels = obj_labels.view(-1, self.num_labels)
                obj_loss = loss_obj(active_obj_logits, active_labels.float())
                obj_loss = obj_loss.view(-1, input_ids.size()[-1], 2)
                obj_loss = torch.mean(obj_loss, 2)
                obj_loss = torch.sum(attention_mask * obj_loss) / torch.sum(attention_mask)
                outputs_obj = (obj_loss,) + obj_outputs
            else:
                # outputs_obj = obj_logits.view(-1,hidden_states.size()[1],self.obj_labels // 2 ,2)
                outputs_obj = active_obj_logits.view(-1, input_ids.size()[-1], 2)
        if obj_train == True:
            return sub_outputs, outputs_obj  # (loss), scores, (hidden_states), (attentions)
        else:
            return sub_outputs

