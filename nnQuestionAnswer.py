import numpy
import sys
from collections import Counter
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.optim as optim

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

#import spacy
import numpy as np

import random
import math
import time

PATH='weights_only.pth'
TRAIN_TEXT_FILE_PATH = 'dialogues.txt'

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

count=0
NUM_SENTENCES=20000
input_sentences = []
output_sentences = []
output_sentences_inputs = []
n=0

# good =pd.read_csv('good.tsv', sep='\t')
# good.sample(3)
# rep= good[good.context_0=='да'].reply
# if rep.shape[0]>0:
#     print(rep.sample(1).iloc[0])



# открытие файла
with open(TRAIN_TEXT_FILE_PATH) as text_file:
    text_sample = text_file.readlines()
text_sample = ' '.join(text_sample)

# разбитие текста на цифру (прямой и обратный соловари)
def text_to_seq(text_sample):
    char_counts = Counter(text_sample)
    char_counts = sorted(char_counts.items(), key = lambda x: x[1], reverse=True)

    sorted_chars = [char for char, _ in char_counts]
    print(sorted_chars)
    char_to_idx = {char: index for index, char in enumerate(sorted_chars)}
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    sequence = np.array([char_to_idx[char] for char in text_sample])
    
    return sequence, char_to_idx, idx_to_char

sequence, char_to_idx, idx_to_char = text_to_seq(text_sample)

# кодер
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        # выходы всегда исходят из верхнего скрытого слоя
        
        return hidden, cell

# декодер
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n направлений в декодере всегда будут равны 1, поэтому:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #последовательность len и n направлений всегда будут равны 1 в декодере, поэтому:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Скрытые размеры кодера и декодера должны быть равны!"
        assert encoder.n_layers == decoder.n_layers, \
            "Кодер и декодер должны иметь одинаковое количество слоев!"        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio - вероятность использования принуждения учителя
        #например, если teacher_forcing_ratio равно 0,75, мы используем вводные данные, основанные на истине, в 75% случаев
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #тензор для хранения выходных данных декодера
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #последнее скрытое состояние кодера используется в качестве начального скрытого состояния декодера
        hidden, cell = self.encoder(src)
        
        #первый ввод в декодер - это токены <sos>
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            #вставка встраивания входного маркера, предыдущих скрытых и предыдущих состояний ячеек
            #получение выходного тензора (предсказаний) и новых скрытых состояний и состояний ячеек
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #поместите прогнозы в тензор, содержащий прогнозы для каждого токена
            outputs[t] = output
            
            #решите, будем ли мы использовать принуждение учителя или нет
            teacher_force = random.random() < teacher_forcing_ratio
            
            #получите самый высокий прогнозируемый токен из наших прогноз
            top1 = output.argmax(1) 
            
            #если учитель принуждает, используйте фактический следующий токен в качестве следующего ввода
            #если нет, используйте предсказанный токен
            input = trg[t] if teacher_force else top1
        
        return outputs

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)


model = Seq2Seq(input_size=len(idx_to_char), hidden_size=128, embedding_size=128, n_layers=2)
model.to(device)

# функция потерь
criterion = nn.CrossEntropyLoss()
# оптимизатор
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)

n_epochs = 50000
loss_avg = []

for epoch in range(n_epochs):
    model.train()
