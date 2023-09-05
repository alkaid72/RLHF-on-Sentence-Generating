# !/usr/bin/env python3
"""

调用本地利用PPO训练好的GPT模型。
Call the GPT model trained locally using PPO.

"""

import torch

from transformers import AutoTokenizer
from trl.gpt2 import GPT2HeadWithValueModel

model_path = 'checkpoints/ppo_sentiment_gpt/model_10_0.87'                  # Model storage address
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2HeadWithValueModel.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.eos_token = tokenizer.pad_token

gen_len = 16
gen_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id
}


def inference(prompt: str):
    """
    根据prompt生成内容。
    Generate content based on prompt.
    
    Args:
        prompt (str): _description_
    """
    inputs = tokenizer(prompt, return_tensors='pt')
    response = model.generate(inputs['input_ids'].to(device),
                                max_new_tokens=gen_len, **gen_kwargs)
    r = response.squeeze()[-gen_len:]
    return tokenizer.decode(r)


if __name__ == '__main__':
    from rich import print

    gen_times = 10                                                  

    prompt = '这部电影很'
    print(f'prompt: {prompt}')
    
    for i in range(gen_times):                      # Generate 10 consecutive answers to the same prompt 
        res = inference(prompt)
        print(f'res {i}: ', res)

