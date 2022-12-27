'''
#!/usr/bin/env python
'''
# import textattack.models.helpers.lstm_for_classification
import torch
# import numpy as np
# from copy import deepcopy
import textattack
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import PyTorchModelWrapper
# from textattack.models.wrappers import ModelWrapper
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
# from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients, LayerDeepLiftShap, InternalInfluence, LayerGradientXActivation
# from captum.attr import visualization as viz
from textattack.attack_recipes import PSOZang2020
from textattack import Attacker
# from textattack.models.wrappers import ModelWrapper
from datasets import load_dataset

name='attacker_PSO'
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(device)

#1. 加载之前从HuggerFace下载好的数据集
dataset = HuggingFaceDataset("rotten_tomatoes", None, "train")

#2. 加载LSTM模型
original_model = textattack.models.helpers.lstm_for_classification.LSTMForClassification.from_pretrained("lstm-mr")
# original_model = textattack.models.helpers.lstm_for_classification.LSTMForClassification(
# max_seq_length=128,
# num_labels=2,
# emb_layer_trainable=True,
# )

#3. 模型包装器
model_wrapper = PyTorchModelWrapper(original_model, original_model.tokenizer)
# original_tokenizer = textattack.models.tokenizers.t5_tokenizer.T5Tokenizer()
# model_wrapper = textattack.models.wrappers.pytorch_model_wrapper.PyTorchModelWrapper(original_model, original_tokenizer)

#4. 搭建攻击方法
attack = textattack.attack_recipes.PSOZang2020.build(model_wrapper)

#5. 传递攻击参数
attack_args = textattack.AttackArgs(num_examples=100,
                                    log_to_csv="{}_log.csv".format(name),
                                    checkpoint_interval=1, checkpoint_dir="{}_check".format(name),
                                    disable_stdout=True)

#6. 启动
attacker = textattack.Attacker(attack, dataset, attack_args)

attacker.attack_dataset()

