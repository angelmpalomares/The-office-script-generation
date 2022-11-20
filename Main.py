import os
from abc import ABC

import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv
torch.cuda.empty_cache()

# Obtain data from the csv and make some processing
data_dir = 'the-office_lines.csv'
data = pd.read_csv(data_dir)
data.Character = data.Character.str.replace(" ", "_")
data.Line = data.Character.astype(str) + ": " + data.Line.astype(str)
clean_data = data[['Line']]

# This class will help tokenize the data
class ScriptLine(Dataset):

    def __init__(self, control_code, truncate=False, gpt2_type="gpt2", max_length=1024):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.line = []

        for row in clean_data.Line:
            self.line.append(torch.tensor(
                self.tokenizer.encode(f"<|{control_code}|>{row[:max_length]}<|endoftext|>")
            ))
        if truncate:
            self.line = self.line[:7500]
        self.line_count = len(self.line)

    def __len__(self):
        return self.line_count

    def __getitem__(self, item):
        return self.line[item]


dataset = ScriptLine(clean_data.Line, truncate=True)

# Accumulated batch size (since GPT2 is so big)
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None


def train(
        training_set, training_model, tokenizer,
        batch_size=24, epochs=6, lr=2e-5,
        max_seq_len=400, warmup_steps=100,
        gpt2_type="gpt2", output_dir=".", output_prefix="wreckgar",
        test_mode=False, save_model_on_epoch=True,
):
    acc_steps = 100
    device = torch.device("cuda:0")
    training_model = training_model.cuda()
    training_model.train()

    optimizer = AdamW(training_model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(training_set, batch_size=1, shuffle=True)
    loss = 0
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):

        print(f"Training epoch {epoch}")
        print(loss)
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = training_model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                training_model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                training_model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    return training_model

# Get the tokenizer and model
GPT2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
GPT2_model = GPT2LMHeadModel.from_pretrained('gpt2')

model = train(dataset, GPT2_model, GPT2_tokenizer)
torch.save(model, "./model.pt")



model = torch.load("./model.pt")
def generate(
        model,
        tokenizer,
        prompt,
        entry_length=500,  # maximum number of words
        top_p=0.9,
        temperature=1,
):
    model.eval()
    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():
        entry_finished = False
        generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

        for i in range(entry_length):
            outputs = model(generated, labels=generated)
            loss, logits = outputs[:2]
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                ..., :-1
                                                ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value

            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

            if next_token in tokenizer.encode("<|endoftext|>"):
                entry_finished = True

            if entry_finished:
                generated_num = generated_num + 1

                output_list = list(generated.squeeze().numpy())
                output_text = tokenizer.decode(output_list)
                generated_list.append(output_text)
                break

        if not entry_finished:
            output_list = list(generated.squeeze().numpy())
            output_text = f"{tokenizer.decode(output_list)}<|endoftext|>"
            generated_list.append(output_text)

    return generated_list


# Function to generate multiple sentences.
def text_generation(texto):
    x = generate(model.to('cpu'), GPT2_tokenizer, texto)[0]
    clean = x.replace("\\", "")
    to_remove = clean.split('.')[-1]
    return clean.replace(to_remove, '')


print(text_generation("Michael: That's what she said"))

