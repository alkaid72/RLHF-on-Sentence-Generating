# !/usr/bin/env python3
"""

测试训练好的奖励模型。
Test the trained reward model.

"""
import torch
from rich import print
from transformers import AutoTokenizer

device = 'cpu'
tokenizer = AutoTokenizer.from_pretrained('./checkpoints/reward_model/sentiment_analysis/model_best/')
model = torch.load('./checkpoints/reward_model/sentiment_analysis/model_best/model.pt')
model.to(device).eval()

texts = [
    '这部电影真的很精彩，剧情出人意料，角色塑造优秀，令人回味',
    '看到一半就睡着了，垃圾电影'
]
inputs = tokenizer(
    texts, 
    max_length=128,
    padding='max_length', 
    return_tensors='pt'
)
r = model(**inputs)
print(r)
