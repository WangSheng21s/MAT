# MAT
We propose Marker LAttice Transformer (MAT), a strong framework for medical IE. This framework is composed of three separate models, each designed for a specific task: medical entity recognition (MER), medical relation extraction (MRE), and medical attribute extraction (MAE). All the models are deeply based on markers embedded in the input text, with which the models compute representations from bottom to top layers.
This allows the representations to encode deep semantic information, leading to better outputs. In addition, we enhance the models by lattice-style incorporation of medical
dictionary information, further pre-training on large-scale EMRs, and auxiliary inputs of medical departments and EMR sections.

## Quick links
* [Overview](#Overview)
* [Setup](#Setup)
  * [Install Dependencies](#Install-dependencies)
  * [Data Preprocessing](#Download-and-preprocess-the-datasets)
  * [Data Format](#Input-data-format)
  * [Trained Models](#Trained-Models)
* [Training Script](#Training-script)
* [Quick Start](#Quick-start)
* [CoNLL03 with Dev](#CoNLL03-with-dev)
* [Citation](#Citation)


## Overview
![](./figs/fig1.png)

In this work, Our main contributions can be summarized as follows:

1. We propose MAT, a strong medical IE framework achieving better performance than previous SOTA relation extraction models. This model is a successful application of markers and utilization of medical dictionary in medical IE tasks. We will release our code and collected dictionary to facilitate future research.

2. We train HwaMei-BERT on large-scale EMRs, and show that it can consistently improve the model performance on downstream IE tasks. HwaMei-BERT will also be publicly released.

3. We employ auxiliary information indicators of medical departments and EMR sections, which effectively introduce global context and further contribute to the model performance.


## Setup
### Install Dependencies

The code is based on huggaface's [transformers](https://github.com/huggingface/transformers) and YeDeming's [PL-Marker](https://github.com/thunlp/PL-Marker). 

Install dependencies and [apex](https://github.com/NVIDIA/apex):
```
pip3 install -r requirement.txt
pip3 install --editable ./transformers
```

### Download and preprocess the datasets
Our experiments are mainly based on [hwamei-500](https://huggingface.co/datasets/FreeJon/hwamei-500) dataset.
In addition, we also conducted experiments on the [CMeEE-V2](https://tianchi.aliyun.com/dataset/177390) and [CCKS-2017](https://www.heywhale.com/mw/dataset/648058405742d97f8f6beca0) datasets.



### Trained Models
We release our pretrained bert-based-hm models on hwamei-datasets on [Hugging face](https://huggingface.co/FreeJon/bert-base-hm-20e-384b-15m)

## Training Script
Download Pre-trained Language Models from [Hugging face](https://huggingface.co/FreeJon/bert-base-hm-20e-384b-15m)
Download Dataset from [hwamei-500](https://huggingface.co/datasets/FreeJon/hwamei-500)
```
mkdir -p bert_models/bert-base-uncased
wget -P bert_models/bert-base-uncased https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin
wget -P bert_models/bert-base-uncased https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt
wget -P bert_models/bert-base-uncased https://huggingface.co/bert-base-uncased/resolve/main/config.json

mkdir -p bert_models/roberta-large
wget -P bert_models/roberta-large https://huggingface.co/roberta-large/resolve/main/pytorch_model.bin
wget -P bert_models/roberta-large https://huggingface.co/roberta-large/resolve/main/merges.txt
wget -P bert_models/roberta-large https://huggingface.co/roberta-large/resolve/main/vocab.json
wget -P bert_models/roberta-large https://huggingface.co/roberta-large/resolve/main/config.json

mkdir -p bert_models/albert-xxlarge-v1
wget -P bert_models/albert-xxlarge-v1 https://huggingface.co/albert-xxlarge-v1/resolve/main/pytorch_model.bin
wget -P bert_models/albert-xxlarge-v1 https://huggingface.co/albert-xxlarge-v1/resolve/main/spiece.model
wget -P bert_models/albert-xxlarge-v1 https://huggingface.co/albert-xxlarge-v1/resolve/main/config.json
wget -P bert_models/albert-xxlarge-v1 https://huggingface.co/albert-xxlarge-v1/resolve/main/tokenizer.json

mkdir -p bert_models/scibert_scivocab_uncased
wget -P bert_models/scibert_scivocab_uncased https://huggingface.co/allenai/scibert_scivocab_uncased/resolve/main/pytorch_model.bin
wget -P bert_models/scibert_scivocab_uncased https://huggingface.co/allenai/scibert_scivocab_uncased/resolve/main/vocab.txt
wget -P bert_models/scibert_scivocab_uncased https://huggingface.co/allenai/scibert_scivocab_uncased/resolve/main/config.json
```

Train NER Models:
```
bash scripts/run_train_ner_PLMarker.sh
bash scripts/run_train_ner_BIO.sh
bash scripts/run_train_ner_TokenCat.sh
```

Train RE Models:
```
bash run_train_re.sh
```

## Quick Start
The following commands can be used to run our pre-trained models on SciERC.

Evaluate the NER model:
```
CUDA_VISIBLE_DEVICES=0  python3  run_acener.py  --model_type bertspanmarker  \
    --model_name_or_path  ../bert_models/scibert-uncased  --do_lower_case  \
    --data_dir scierc  \
    --learning_rate 2e-5  --num_train_epochs 50  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 2000  --max_pair_length 256  --max_mention_ori_length 8    \
    --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed 42  --onedropout  --lminit  \
    --train_file train.json --dev_file dev.json --test_file test.json  \
    --output_dir sciner_models/sciner-scibert  --overwrite_output_dir  --output_results
```

We need the ner result `ent_pred_test.json` from the NER model with `--output_results`. Then we evaluate the RE model:
```
CUDA_VISIBLE_DEVICES=0  python3  run_re.py  --model_type bertsub  \
    --model_name_or_path  ../bert_models/scibert-uncased  --do_lower_case  \
    --data_dir scierc  \
    --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 256  --max_pair_length 16  --save_steps 2500  \
    --do_eval  --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax  \
    --fp16   \
    --test_file sciner_models/sciner-scibert/ent_pred_test.json  \
    --use_ner_results \
    --output_dir scire_models/scire-scibert
```
Here,  `--use_ner_results` denotes using the original entity type predicted by NER models.


## CoNLL03 with Dev

| Model | Original split | Train with dev set |
| :-----| :----: | :----: |
| SeqTagger | 93.6 | 93.9 |
| T-Concat  | 93.0 | 93.3 |
| PL-Marker | 94.0 | 94.2 |


## Citation
If you use our code in your research, please cite our work:
```bibtex
@inproceedings{ye2022plmarker,
  author    = {Deming Ye and
               Yankai Lin and
               Peng Li and
               Maosong Sun},
  editor    = {Smaranda Muresan and
               Preslav Nakov and
               Aline Villavicencio},
  title     = {Packed Levitated Marker for Entity and Relation Extraction},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational
               Linguistics (Volume 1: Long Papers), {ACL} 2022, Dublin, Ireland,
               May 22-27, 2022},
  pages     = {4904--4917},
  publisher = {Association for Computational Linguistics},
  year      = {2022},
  url       = {https://aclanthology.org/2022.acl-long.337},
  timestamp = {Wed, 18 May 2022 15:21:43 +0200},
  biburl    = {https://dblp.org/rec/conf/acl/YeL0S22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
