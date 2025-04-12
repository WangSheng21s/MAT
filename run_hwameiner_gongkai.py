# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
from collections import defaultdict
import re
import shutil
import time

import code

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                          BertTokenizer,
                          RobertaConfig,
                          RobertaTokenizer,
                          get_linear_schedule_with_warmup,
                          AdamW,
                          BertForNER,
                          BertForSpanNER,
                          BertForSpanMarkerNER,
                          BertForSpanMarkerBiNER,
                          AlbertForNER,
                          AlbertConfig,
                          AlbertTokenizer,
                          BertForLeftLMNER,
                          RobertaForNER,
                          RobertaForSpanNER,
                          RobertaForSpanMarkerNER,
                          AlbertForSpanNER,
                          AlbertForSpanMarkerNER,
                          )

from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, Dataset
import json
import pickle
import numpy as np
import unicodedata
import itertools
import math
from tqdm import tqdm
import re
import timeit

import dic

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForNER, BertTokenizer),
    'bertspan': (BertConfig, BertForSpanNER, BertTokenizer),
    'bertspanmarker': (BertConfig, BertForSpanMarkerNER, BertTokenizer),
    'bertspanmarkerbi': (BertConfig, BertForSpanMarkerBiNER, BertTokenizer),
    'bertleftlm': (BertConfig, BertForLeftLMNER, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForNER, RobertaTokenizer),
    'robertaspan': (RobertaConfig, RobertaForSpanNER, RobertaTokenizer),
    'robertaspanmarker': (RobertaConfig, RobertaForSpanMarkerNER, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertForNER, AlbertTokenizer),
    'albertspan': (AlbertConfig, AlbertForSpanNER, AlbertTokenizer),
    'albertspanmarker': (AlbertConfig, AlbertForSpanMarkerNER, AlbertTokenizer),
}


departments_label = ['gAAAAABkJOwVGx_1aZ9UsdGQFG3m7BZEn1kugLAwRcENw3Zd--D4ejw_DWfSEAqHpqYS5GNyHAMahv99R5SAMqfKzeCMpB26MQ==', 'gAAAAABkJOwVZJ4T5tIwpaLhlgg7vF_EnrchYWcAPDqIh8AuAZsZV_MLzMWKIMnq4jzRIG-s3LwmHSofdx6bz9lLgDsYPiGHcg==', 'gAAAAABkJOwVpQkBY1vYmmmej5L7-lkDSyueKXWGH45x_I6HDSfQvZWbpaC52PpRVA_Q-ojC3VClVQR6yVsVXuZ4IVsM0qQrVw==', 'gAAAAABkJOwVhjivEDrsEsxlvVQt4eJNjFxaPsOGduicH_6dl7tTmIIMNaMhDTyVXdjhItEGggpX7lwsv2JeP862QHfrnoVdHQ==', 'gAAAAABkJOwVHkUdQARbCN4nEd47OYC7jNCWmg-GV0T_4Qg1OrUAhVCcj9d04VYWLUHvS5LHBa27n8xw1SP7dTq39v0RtksiwA==', 'gAAAAABkJOwVmqfXTwzCCIu_xWjK7wgtrwypfVGFxSfwWYjBSuWZi0y9oJIXex6m-fH6nUxRj8MvSHZiDtMaIjdSwUVvHgA0nA==', 'gAAAAABkJOwUZuHG9C_B1t8_OIbsgRvQH9o_N9Cni3yNGcGV5eL7XOQNoy-Mr7MspXVD_gUS6tzniJxPaBN2CqDnVrX6jvVD2A==', 'gAAAAABkJOwVqJyOngsuUnXc0ZjitTdeF3oZD7iLgbOfZh8GndK5WjoZxW3erGlx4XpqvfNdlztGo2H34COGZiVnKDP4gAJS_A==', 'gAAAAABkJOwVSbkxtZuXjl_9Ihp5-y561zDute4h8xRlTmjROZWBgU_5Wzp0aabi8c_U0gSZ_rk3aEqetK3amArrG5TZoH85iQ==', 'gAAAAABkJOwURECI8L5XhuVEvjVNoY7MSraXXQwkHdFr1KwXMg9czJieHJp15Hj8Jd0zJETcZOTzI0IjuJc6Z83wUnIUcJBfGg==', 'gAAAAABkJOwUn9e_DnzDjpYQIcwIMKq0z61S9YfU5hD6bHHCbbQ_bpoK3wnvAcstyMqZLSEaWDitE1-ndgxKdITmkMzO85yGcg==', 'gAAAAABkJOwVUiTmVLgOccAYSdK9-_LVjQ_6GP6Ogm0otktZHWGOwRcV99GR4DuU5GMrt_PUM9MkrIJTJ_4yAu1FpYscleBhGw==', 'gAAAAABkJOwVnzF8rNOkzgFheIeIDWIWg_SHKE5I9G1RLae316_YWMKLvqRVNTk5EtDEPBWBMVeC46i17ecEVxaELJudCixFMQ==', 'gAAAAABkJOwU2HXSE5eq85Rhe9IRKeZGVSVGmVCkznGAitLhnTPcQK-wwiVPi0_EriMY1-LRrVfR9j6GGcM6myH2HAS1uQ_-Qw==', 'gAAAAABkJOwVUWqL5Oe70bB78ZmPlE4Sh7kyxLKEnimOsN_K_2H7YKxl9z_5xx-DU_XXnLzee359SifmYM4T1lABMJbhbBG6AA==', 'gAAAAABkJOwVsUTgURIx7bNj9N1Gcet-kCSXFKtn751faQgbsFurpwsM6jMTwYYQ7_juc2i01CHCU5bJ8ZuYlL1ll_BbfMcW5A==', 'gAAAAABkJOwUvRYNmPbhvtAhjgKXv3Uk1u_i6zY-rFO4Xxuew5YGwpAz58wYGGP_B1FInfJw-pxV3wfHy4R58cYvTJxssl3YyA==', 'gAAAAABkJOwU7e2R_FrRmimT_pt-vS6flgo9v9SMuueE1fVmgeXs7AF-uXhmrABmgCqPcq-j1nQaObN6w5U50NvffKb17EsY3g==', 'gAAAAABkJOwUaT3FPq5vy6EzsMiqDiPd4np0-O2ziSiJuRQRzkFJgIDLC-YNChjuvdDLuTAvcMy1YCkVBtLojEbqtsrPHewndQ==', 'gAAAAABkJOwV0xIKWHzOF93v90ys9UFwtHbiojbhyAYAjeNU-sBvt7eIX0qMT2w2StkIA7l_1UCq-V5Xem6X_CmBWq2n_vO1Ow==', 'gAAAAABkJOwUgDoEeeR4srw3l72cOHLetSUOgJwQrGokO4MHpKCfnUnR1CylCn3eK15Ban09ubnwI7ytHY9FszMNhWBaLi4lFQ==', 'gAAAAABkJOwVaPfdxlmPNwaKMw2hiklQwCg9xpZrULNLatM3biEVZa-XtQbF6QvolJ5-hSuNquDM8fyly0hPR1f93wUL5_odeQ==', 'gAAAAABkJOwVYaIUdUA7n_Rl-7sOoR7xIFAMqWcUY9gL1S9T3IKYjqc9EUlWKqZ9sFa4ya6WTfA3Kqxu75GMaQmlwPqF7NAumQ==', 'gAAAAABkJOwVmDoookP7XrJPARs2h9w6NgE1a-gV0vk0ka4wjo4NsTYp9fw9Pf8xSL9S44PlA-g80WMh83WvCbq8Qxoj-ki6vQ==', 'gAAAAABkJOwVD8_9Qllsb7nQZtk-eI8SyiAwMdi9_kqOxw-D9Ub5EgioTac-czMUwZAza0ohfRIqWGR0-5EPRanaoqfJ9_-ohg==', 'gAAAAABkJOwV0R_-lBZ5-ZhW4rGHIBeWyvA9-JmETcW1G9jS8yNiHyfl-JyukWI6OtlCDi6UR_C3tF-mGMPTJBGeZaOfOtpTBw==', 'gAAAAABkJOwUJSaC3mQ-zmXcxgUkGTUMKjdJLnzkkyzKTmPB7YusvfnaJNNMOUmteAWe5j0H9ULy9Dln5Bu96T1H441hgiiRLw==', 'gAAAAABkJOwVH0g4l3CAyTpl_s3rudLlnN8xYHvlk819SiOsuQ5SjWWa6TGpnqh6hBlwpDxZqtEk0UeMWWDtibrrNM7yW6WUhQ==', 'gAAAAABkJOwUMNQ6bjbCtA5XRCbfyKGftgXySXMLpW7HXEdVv0vED_2VaGJgXeg7GoBuqAdX1HwdOXPlGeT_Cv1w-YHW_sSvSg==', 'gAAAAABkJOwVy59m8Ve4lVou6bXAMtXBmT-de2-7e23b4pl5UvkZQBnaQ2RZTd7V7iPFxZmqJawzQblmiwozikphSFI7KRF0ug==', 'gAAAAABkJOwVczKypEp1N7YWpEF-GYe4YyInSImXrYbj67SkoA6S_7im8DwQRVVfL-zYPf5Ng3n53zXYs3iCAN34OK89NL_ECg==', 'gAAAAABkJOwVwEHezZoGSNV6BphXUo0ZVdfDgQoGCnO-DOD4DaY7Jwnu2Jy7PfiiFniY1Vgbb-PmtJREhGuBnDnkqLtX1zxd_w==', 'gAAAAABkJOwV5lB_1oyOSrWMFvBfICsU2zaQ-Zq_BdZJ2uGQBsEHCoYkXxADD97Pc-SoMSxw9UaVWQHJpbDrBZZg5DiKyCcXAQ==', 'gAAAAABkJOwULAhnKpjc9kr89sx1qFc0XcKq47Qqhgl626eyHGvD87R98nOYB5PPpAOhnwE0x2CS8rUEaqSuBOJFHxrxCYUolQ==', 'gAAAAABkJOwU-QNbBt2n0zknybcPiyqjSKGRAmf8PtGgnZuRFqX2BbQSC-DVPDeuBT5T1pwLNeakNOQJCTsGsRyOmb1Iqwalzg==', 'gAAAAABkJOwV_gZ6jcIf21kAKvZBWdmU7sybf5jV0DiHCEJasbjawwKRWU7lyA1iAZIoY99IM3fdjWP3PJQWhh4LBdyvb7F67A==', 'gAAAAABkJOwUvnMyiUEgOWEFMz3NcjvVOUTcmmpAtD0B846JeYcZpa0Cv8bQH1bA_S3g4rBvdYoyQYVPLlWvjIJfDpdb-gzdcg==']


departments_label_map  = {departments_label[i]:i for i in range(len(departments_label))}


class HwaMeiDatasetNER(Dataset):
    def __init__(self, tokenizer, args=None, evaluate=False, do_test=False):
        if not evaluate:
            file_path = os.path.join(args.data_dir, args.train_file)
        else:
            if do_test:
                file_path = os.path.join(args.data_dir, args.test_file)
            else:
                file_path = os.path.join(args.data_dir, args.dev_file)

        assert os.path.isfile(file_path)

        self.file_path = file_path

        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length

        self.evaluate = evaluate
        self.local_rank = args.local_rank
        self.args = args
        self.model_type = args.model_type

        self.ner_label_list = ['NIL','Self_Reported_Abnormality', 'Test_Result', 'Test_Process', 'Drug', 'Disease_or_Syndrome', 'Abnormal_Test_Result','Operation','Body_Part','Equipment','Drug_Dose','Prevention','Treatment','Care','Injury_or_Poisoning','Department','Organ_Damage','Personal_History','Body_Matter']

        self.ci_use_diff = args.ci_use_diff
        self.add_cixinxi = args.add_cixinxi
        if self.add_cixinxi:  # 需要引入词信息
            self.ci_file_paths = ['../ciku/yixueciku/THUOCL_medical.txt', '../ciku/yixueciku/body中文身体部位名称.txt',
                                  '../ciku/yixueciku/disease_new.txt', '../ciku/yixueciku/ICD10诊断.scel',
                                  '../ciku/yixueciku/symptom.txt', '../ciku/yixueciku/部分疾病名药名.scel',
                                  '../ciku/yixueciku/西医病名.scel', '../ciku/yixueciku/医院电子病历词库.scel',
                                  '../ciku/yixueciku/症状.scel']
            self.cidian = dic.get_tree_find(self.ci_file_paths)

        self.max_pair_length = args.max_pair_length

        self.max_entity_length = args.max_pair_length * 2
        self.initialize()

    def is_punctuation(self, char):
        # obtained from:
        # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def get_original_token(self, token):
        escape_to_original = {
            "-LRB-": "(",
            "-RRB-": ")",
            "-LSB-": "[",
            "-RSB-": "]",
            "-LCB-": "{",
            "-RCB-": "}",
        }
        if token in escape_to_original:
            token = escape_to_original[token]
        return token


    def initialize(self):
        tokenizer = self.tokenizer
        max_num_subwords = self.max_seq_length - 2

        ner_label_map = {label: i for i, label in enumerate(self.ner_label_list)}

        def tokenize_word(text):
            if (
                    isinstance(tokenizer, RobertaTokenizer)
                    and (text[0] != "'")
                    and (len(text) != 1 or not self.is_punctuation(text))
            ):
                return tokenizer.tokenize(text, add_prefix_space=True)
            return tokenizer.tokenize(text)

        f = open(self.file_path, "r", encoding='utf-8')
        self.data = []
        self.tot_recall = 0
        self.ner_golden_labels = set([])

        for l_idx, line in enumerate(f):
            data = json.loads(line)
            # if len(self.data) > 5:
            #     break

            if self.args.output_dir.find('test') != -1:
                if len(self.data) > 5:
                    break

            tokens = data['tokens']

            ners = data['entities']
            department = data['department']
            secction_id = data['section_id']

            department_label = departments_label_map[department]
            weizhi_label = secction_id

            num_ners = len(ners)
            self.tot_recall += num_ners
            entities = []
            for i in range(len(ners)):
                ner = ners[i]
                entities.append((ner['start'], ner['end'], ner['type']))

            words = tokens

            entitity_labels = {}

            for start, end, label in entities:
                entitity_labels[(start, end)] = ner_label_map[label]
                self.ner_golden_labels.add(((l_idx, 0), (start,end), label))

            entity_infos = []

            for entity_start in range(len(words)):
                for entity_end in range(entity_start+1, len(words)):

                    if entity_end - entity_start + 1 >self.args.max_mention_ori_length:
                        break

                    label = entitity_labels.get((entity_start, entity_end), 0)

                    cihui = words[entity_start: entity_end]
                    if self.add_cixinxi and self.cidian.find(cihui):#cihui是一个词，那么加入cihuibiao中用于后续处理
                        entity_infos.append((entity_start, entity_end, label, True))#增加了一个bool标记，用来标记这个span是否是词汇
                    else:
                        entity_infos.append((entity_start, entity_end, label, False))



            dL = self.max_pair_length
            if self.args.shuffle:
                random.shuffle(entity_infos)
            if self.args.group_sort:
                group_axis = np.random.randint(2)
                sort_dir = bool(np.random.randint(2))
                entity_infos.sort(key=lambda x: (x[0][group_axis], x[0][1 - group_axis]), reverse=sort_dir)

            if not self.args.group_edge:
                for i in range(0, len(entity_infos), dL):
                    examples = entity_infos[i: i + dL]
                    item = {
                        'sentence': words,
                        'weizhi_label': weizhi_label,
                        'department_label': department_label,
                        'examples': examples,
                        'example_index': (l_idx, 0),
                        'example_L': len(entity_infos)
                    }
                    self.data.append(item)
            else:
                if self.args.group_axis == -1:
                    group_axis = np.random.randint(2)
                else:
                    group_axis = self.args.group_axis
                sort_dir = bool(np.random.randint(2))
                entity_infos.sort(key=lambda x: (x[0][group_axis], x[0][1 - group_axis]), reverse=sort_dir)
                _start = 0
                while _start < len(entity_infos):
                    _end = _start + dL
                    if _end >= len(entity_infos):
                        _end = len(entity_infos)
                    else:
                        while entity_infos[_end - 1][0][group_axis] == entity_infos[_end][0][
                            group_axis] and _end > _start:
                            _end -= 1
                        if _start == _end:
                            _end = _start + dL

                    examples = entity_infos[_start: _end]

                    item = {
                        'sentence': words,
                        'weizhi_label': weizhi_label,
                        'department_label': department_label,
                        'examples': examples,
                        'example_index': (l_idx, 0),
                        'example_L': len(entity_infos)
                    }

                    self.data.append(item)
                    _start = _end

        # exit()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        weizhi_label = entry['weizhi_label']
        department_label = entry['department_label']

        input_ids = self.tokenizer.convert_tokens_to_ids(entry['sentence'])
        L = len(input_ids)

        input_ids += [0] * (self.max_seq_length - len(input_ids) )
        position_plus_pad = int(self.model_type.find('roberta') != -1) * 2

        if self.model_type not in ['bertspan', 'robertaspan', 'albertspan']:

            if self.model_type.startswith('albert'):
                input_ids = input_ids + [30000] * (len(entry['examples'])) + [0] * (
                            self.max_pair_length - len(entry['examples']))
                input_ids = input_ids + [30001] * (len(entry['examples'])) + [0] * (
                            self.max_pair_length - len(entry['examples']))
            elif self.model_type.startswith('roberta'):
                input_ids = input_ids + [50261] * (len(entry['examples'])) + [0] * (
                            self.max_pair_length - len(entry['examples']))
                input_ids = input_ids + [50262] * (len(entry['examples'])) + [0] * (
                            self.max_pair_length - len(entry['examples']))
            else:
                input_ids = input_ids + [1] * (len(entry['examples'])) + [0] * (
                            self.max_pair_length - len(entry['examples']))
                input_ids = input_ids + [2] * (len(entry['examples'])) + [0] * (
                            self.max_pair_length - len(entry['examples']))
            input_ids = input_ids + [3] * 2#留了两个marker的位置给华美数据集用来增加departmernt信息和电子病历位置信息

            attention_mask = torch.zeros(
                (self.max_entity_length + self.max_seq_length + 2, self.max_entity_length + self.max_seq_length + 2),
                dtype=torch.int64)
            attention_mask[:L, :L] = 1
            if self.args.add_binglixinxi2 == 1 or self.args.add_binglixinxi2 == 3:
                attention_mask[:L, -1] = 1
            if self.args.add_binglixinxi2 == 2 or self.args.add_binglixinxi2 == 3:
                attention_mask[:L, -2] = 1
            position_ids = list(range(position_plus_pad, position_plus_pad + self.max_seq_length)) + [
                0] * self.max_entity_length + [0] * 2

        else:
            attention_mask = [1] * L + [0] * (self.max_seq_length - L)
            attention_mask = torch.tensor(attention_mask, dtype=torch.int64)
            position_ids = list(range(position_plus_pad, position_plus_pad + self.max_seq_length)) + [
                0] * self.max_entity_length

        labels = []
        mentions = []
        mention_pos = []
        num_pair = self.max_pair_length

        full_attention_mask = [1] * L + [0] * (self.max_seq_length - L) + [0] * (self.max_pair_length) * 2 + [0] * 2


        position_ids[-1] = 511
        position_ids[-2] = 510#最后两个额外的marker绑定给位置0
        attention_mask[-2:, :L] = 1
        department_em = '[unused%d]' % (30 + department_label)
        weizhi_em = '[unused%d]' % (70 + weizhi_label)
        input_ids[-2] = self.tokenizer.convert_tokens_to_ids(department_em)
        input_ids[-1] = self.tokenizer.convert_tokens_to_ids(weizhi_em)

        for x_idx, x in enumerate(entry['examples']):
            start = x[0]
            end = x[1]
            label = x[2]
            if_cihui = x[3]
            mentions.append((start, end))
            mention_pos.append((start, end))
            labels.append(label)

            if self.model_type in ['bertspan', 'robertaspan', 'albertspan']:
                continue

            w1 = x_idx
            w2 = w1 + num_pair

            w1 += self.max_seq_length
            w2 += self.max_seq_length
            position_ids[w1] = start
            position_ids[w2] = end


            for xx in [w1, w2]:
                full_attention_mask[xx] = 1
                if self.add_cixinxi and if_cihui:  # 如果该span是词汇并且需要引入词汇信息
                    attention_mask[:L, xx] = 1
                    if self.ci_use_diff:
                        input_ids[xx] = input_ids[xx] + 2#将普通的span与词对应的span区分开来
                for yy in [w1, w2]:
                    attention_mask[xx, yy] = 1
                attention_mask[xx, :L] = 1





        labels += [-1] * (num_pair - len(labels))
        mention_pos += [(0, 0)] * (num_pair - len(mention_pos))

        item = [torch.tensor(input_ids),
                attention_mask,
                torch.tensor(position_ids),
                torch.tensor(labels, dtype=torch.int64),
                torch.tensor(mention_pos),
                torch.tensor(full_attention_mask)
                ]

        if self.evaluate:
            item.append(entry['example_index'])
            item.append(mentions)
        
        #code.interact(local=locals())

        return item

    @staticmethod
    def collate_fn(batch):
        fields = [x for x in zip(*batch)]

        num_metadata_fields = 2
        stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
        stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors

        return stacked_fields


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def train(args, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        # tb_writer = SummaryWriter("logs/ace_ner_logs/"+args.output_dir[args.output_dir.rfind('/'):])
        tb_writer = SummaryWriter(
            "logs/" + args.data_dir[max(args.data_dir.rfind('/'), 0):] + "_ner_logs/" + args.output_dir[
                                                                                        args.output_dir.rfind('/'):])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = HwaMeiDatasetNER(tokenizer=tokenizer, args=args)
    #data_0 = train_dataset[0]#用于调试
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  num_workers= 8, pin_memory = True) #2 * int(args.output_dir.find('test') == -1))

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_steps == -1:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * t_total), num_training_steps=t_total
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # ori_model = model
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    best_f1 = -1

    for _ in train_iterator:
        # if _ > 0 and (args.shuffle or args.group_edge or args.group_sort):
        #     train_dataset.initialize()
        #     if args.group_edge:
        #         train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        #         train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=2*int(args.output_dir.find('test')==-1))

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'position_ids': batch[2],
                      'labels': batch[3],
                      }

            if args.model_type.find('span') != -1:
                inputs['mention_pos'] = batch[4]
            if args.use_full_layer != -1:
                inputs['full_attention_mask'] = batch[5]

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            #print(loss)
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    update = True
                    # Save model checkpoint
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        f1 = results['f1']
                        tb_writer.add_scalar('f1', f1, global_step)

                        if f1 > best_f1:
                            best_f1 = f1
                            print('Best F1', best_f1)
                        else:
                            update = False

                    if update:
                        checkpoint_prefix = 'checkpoint'
                        output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        _rotate_checkpoints(args, checkpoint_prefix)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_f1


def evaluate(args, model, tokenizer, prefix="", do_test=False):
    eval_output_dir = args.output_dir

    results = {}

    eval_dataset = HwaMeiDatasetNER(tokenizer=tokenizer, args=args, evaluate=True, do_test=do_test)
    ner_golden_labels = set(eval_dataset.ner_golden_labels)
    ner_tot_recall = eval_dataset.tot_recall

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)

    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=HwaMeiDatasetNER.collate_fn,
                                 num_workers=4 * int(args.output_dir.find('test') == -1))

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    scores = defaultdict(dict)
    predict_ners = defaultdict(list)

    model.eval()

    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        indexs = batch[-2]
        batch_m2s = batch[-1]

        batch = tuple(t.to(args.device) for t in batch[:-2])

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'position_ids': batch[2],
                      #   'labels':         batch[3]
                      }

            if args.model_type.find('span') != -1:
                inputs['mention_pos'] = batch[4]
            if args.use_full_layer != -1:
                inputs['full_attention_mask'] = batch[5]

            outputs = model(**inputs)

            ner_logits = outputs[0]
            ner_logits = torch.nn.functional.softmax(ner_logits, dim=-1)
            #print(ner_logits)
            ner_values, ner_preds = torch.max(ner_logits, dim=-1)

            for i in range(len(indexs)):
                index = indexs[i]
                m2s = batch_m2s[i]
                for j in range(len(m2s)):
                    obj = m2s[j]
                    ner_label = eval_dataset.ner_label_list[ner_preds[i, j]]
                    if ner_label != 'NIL':
                        scores[(index[0], index[1])][(obj[0], obj[1])] = (float(ner_values[i, j]), ner_label)

    cor = 0
    tot_pred = 0
    cor_tot = 0
    tot_pred_tot = 0

    for example_index, pair_dict in scores.items():

        sentence_results = []
        for k1, (v2_score, v2_ner_label) in pair_dict.items():
            if v2_ner_label != 'NIL':
                sentence_results.append((v2_score, k1, v2_ner_label))

        sentence_results.sort(key=lambda x: -x[0])
        no_overlap = []

        def is_overlap(m1, m2):
            if m2[0] <= m1[0] and m1[0] <= m2[1]:
                return True
            if m1[0] <= m2[0] and m2[0] <= m1[1]:
                return True
            return False

        for item in sentence_results:
            m2 = item[1]
            overlap = False
            for x in no_overlap:
                _m2 = x[1]
                if (is_overlap(m2, _m2)):
                    if args.data_dir.find('ontonotes') != -1:
                        overlap = True
                        break
                    else:

                        if item[2] == x[2]:
                            overlap = True
                            break

            if not overlap:
                no_overlap.append(item)

            pred_ner_label = item[2]
            tot_pred_tot += 1
            if (example_index, m2, pred_ner_label) in ner_golden_labels:
                cor_tot += 1

        for item in no_overlap:
            m2 = item[1]
            pred_ner_label = item[2]
            tot_pred += 1
            if args.output_results:
                predict_ners[example_index].append((m2[0], m2[1], pred_ner_label))
            if (example_index, m2, pred_ner_label) in ner_golden_labels:
                cor += 1
        #print("sentence_results为：",sentence_results)
        #print("no_overlap为：", no_overlap)
        #print("tot_predwei:",tot_pred,cor,ner_tot_recall)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f example per second)", evalTime, len(eval_dataset) / evalTime)

    precision_score = p = cor / tot_pred if tot_pred > 0 else 0
    recall_score = r = cor / ner_tot_recall
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0

    p = cor_tot / tot_pred_tot if tot_pred_tot > 0 else 0
    r = cor_tot / ner_tot_recall
    f1_tot = 2 * (p * r) / (p + r) if cor > 0 else 0.0

    results = {'f1': f1, 'f1_overlap': f1_tot, 'precision': precision_score, 'recall': recall_score}

    logger.info("Result: %s", json.dumps(results))

    if args.output_results:
        f = open(eval_dataset.file_path, encoding='utf-8')
        if do_test:
            output_w = open(os.path.join(args.output_dir, 'ent_pred_test.json'), "w", encoding="utf-8")
        else:
            output_w = open(os.path.join(args.output_dir, 'ent_pred_dev.json'), "w", encoding="utf-8")
        for l_idx, line in enumerate(f):
            data = json.loads(line)
            num_sents = len(data['tokens'])
            predicted_ner = []
            for n in range(num_sents):
                item = predict_ners.get((l_idx, n), [])
                item.sort()
                predicted_ner.append(item)

            data['predicted_ner'] = predicted_ner
            output_w.write(json.dumps(data, ensure_ascii=False) + '\n')

    return results


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='hwamei_500', type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run test on the dev set.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=-1, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=5,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--save_total_limit', type=int, default=1,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')

    parser.add_argument("--train_file", default="train.json", type=str)
    #parser.add_argument("--train_file", default="dev.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)

    parser.add_argument('--alpha', type=float, default=1, help="")
    parser.add_argument('--max_pair_length', type=int, default=256, help="")
    parser.add_argument('--max_mention_ori_length', type=int, default=8, help="")
    parser.add_argument('--lminit', action='store_true')
    parser.add_argument('--norm_emb', action='store_true')
    parser.add_argument('--output_results', action='store_true')
    parser.add_argument('--onedropout', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--use_full_layer', type=int, default=-1, help="")
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--group_edge', action='store_true')
    parser.add_argument('--group_axis', type=int, default=-1, help="")
    parser.add_argument('--group_sort', action='store_true')

    parser.add_argument('--add_cixinxi', action='store_true', help="通过搭建医学词表，从而针对性的引入词信息")
    parser.add_argument('--ci_use_diff', action='store_true', help="之前所有的span所对应的marker初始化好像都是一样的，但是感觉有了词信息之后，我们可以将词的marker初始化为不同的初始值")
    parser.add_argument('--continue_train_from_saved_model', type=str, default='', help="加载之前训练过的模型继续进行训练")
    parser.add_argument('--add_binglixinxi', type=int, default=0, help="默认情况下为0,=表示没有额外引入信息，为1时表示引入department信息和weizhixinxi1，为2时表示引入department信息和weizhixinxi2")
    parser.add_argument('--add_binglixinxi2',type=int, default=0, help="默认情况下为0，表示病历marker可以看到token但是token看不到病历marker，为1时表示两者互相可以看到")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    def create_exp_dir(path, scripts_to_save=None):
        if args.output_dir.endswith("test"):
            return
        if not os.path.exists(path):
            os.mkdir(path)

        print('Experiment dir : {}'.format(path))
        if scripts_to_save is not None:
            if not os.path.exists(os.path.join(path, 'scripts')):
                os.mkdir(os.path.join(path, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(path, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)

    if args.do_train and args.local_rank in [-1, 0] and args.output_dir.find('test') == -1:
        create_exp_dir(args.output_dir,
                       scripts_to_save=['run_acener.py', 'transformers/src/transformers/modeling_bert.py',
                                        'transformers/src/transformers/modeling_albert.py'])

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)
    num_labels = 19#hwamei数据集中一共有18个实体类别，加上NIL一共19个

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    config.max_seq_length = args.max_seq_length
    config.alpha = args.alpha
    config.onedropout = args.onedropout
    config.use_full_layer = args.use_full_layer
    config.binglixinxi2 = args.add_binglixinxi2
    if args.continue_train_from_saved_model != '':
        model = model_class.from_pretrained(args.continue_train_from_saved_model, config=config)
    else:
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)

    if args.model_type.startswith('albert'):
        special_tokens_dict = {'additional_special_tokens': ['[unused' + str(x) + ']' for x in range(4)]}
        tokenizer.add_special_tokens(special_tokens_dict)
        # print ('add tokens:', tokenizer.additional_special_tokens)
        # print ('add ids:', tokenizer.additional_special_tokens_ids)
        model.albert.resize_token_embeddings(len(tokenizer))

    if args.do_train and args.lminit:
        if args.model_type.find('roberta') == -1:
            entity_id = tokenizer.encode('entity', add_special_tokens=False)
            assert (len(entity_id) == 1)
            entity_id = entity_id[0]
            mask_id = tokenizer.encode('[MASK]', add_special_tokens=False)
            assert (len(mask_id) == 1)
            mask_id = mask_id[0]
        else:
            entity_id = 10014
            mask_id = 50264

        logger.info('entity_id: %d', entity_id)
        logger.info('mask_id: %d', mask_id)

        if args.model_type.startswith('albert'):
            word_embeddings = model.albert.embeddings.word_embeddings.weight.data
            word_embeddings[30000].copy_(word_embeddings[mask_id])
            word_embeddings[30001].copy_(word_embeddings[entity_id])
        elif args.model_type.startswith('roberta'):
            word_embeddings = model.roberta.embeddings.word_embeddings.weight.data
            word_embeddings[50261].copy_(word_embeddings[mask_id])  # entity
            word_embeddings[50262].data.copy_(word_embeddings[entity_id])
        else:
            word_embeddings = model.bert.embeddings.word_embeddings.weight.data
            word_embeddings[1].copy_(word_embeddings[mask_id])
            word_embeddings[2].copy_(word_embeddings[entity_id])  # entity

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_f1 = 0
    # Training
    if args.do_train:
        global_step, tr_loss, best_f1 = train(args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        update = True
        if args.evaluate_during_training:
            results = evaluate(args, model, tokenizer)
            f1 = results['f1']
            if f1 > best_f1:
                best_f1 = f1
                print('Best F1', best_f1)
            else:
                update = False

        if update:
            checkpoint_prefix = 'checkpoint'
            output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training

            model_to_save.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
            _rotate_checkpoints(args, checkpoint_prefix)

        tokenizer.save_pretrained(args.output_dir)

        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluation
    results = {'dev_best_f1': best_f1}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]

        WEIGHTS_NAME = 'pytorch_model.bin'

        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))

        logger.info("Evaluate on test set")

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""

            model = model_class.from_pretrained(checkpoint, config=config)

            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step, do_test=not args.no_test)

            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.local_rank in [-1, 0]:
        output_eval_file = os.path.join(args.output_dir, "results.json")
        json.dump(results, open(output_eval_file, "w", encoding='utf-8'), ensure_ascii=False)
        logger.info("Result: %s", json.dumps(results))


if __name__ == "__main__":
    main()



