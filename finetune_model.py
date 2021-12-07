import os
from transformers import BertTokenizerFast, BertModel, BertConfig, BertForTokenClassification, TextDataset
from transformers import AdamW,get_linear_schedule_with_warmup, PreTrainedTokenizer, PreTrainedTokenizerFast, BertForSequenceClassification,get_cosine_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup, Trainer, DataCollatorForTokenClassification, AdamWeightDecay, get_cosine_with_hard_restarts_schedule_with_warmup
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
import torch
from tqdm.notebook import tqdm
import os
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score
import json
from collections import Counter
from sklearn.metrics import classification_report
from tqdm import tqdm
import argparse
from dataclasses import dataclass, field
from typing import Optional
import torch.nn as nn
from sklearn.model_selection import  train_test_split
import random
import re
import argparse
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import pickle as pkl
import argparse

def load_checkpoint_cls(model_name, path):
    
    model = BertForSequenceClassification.from_pretrained(model_name)
    state_dict = torch.load(path)

    mkeys = list(model.state_dict().keys())
    skeys = list(state_dict.keys())


    for k in skeys:

        if k not in mkeys:
            del state_dict[k]

    mkeys = list(model.state_dict().keys())
    skeys = list(state_dict.keys())

    for k in mkeys:

        if k not in skeys:
            state_dict[k] = model.state_dict()[k]


    model.load_state_dict(state_dict)
    
    
    return model




parser = argparse.ArgumentParser(description='Train Binary Classifier')
parser.add_argument('--model', type=str, default="deepset/gbert-large")
parser.add_argument('--target', type=str)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learningrate', type=float, default=2e-5)
args = parser.parse_args()

model = load_checkpoint_cls(args.model, "/home/ext/konle/sentiment-datasets/bin_cls/models/eval_lyrik_checkpoint_500.pt")

emotion = args.target

def inner_shuffle_train(X_train, y_train, bsize, state):

    frame = pd.DataFrame(X_train)
    frame["Y"] = y_train
    frame = frame.sample(frac=1,random_state=state)
    class_size = min(Counter(y_train).values())
    
    sel = pd.concat([frame[frame["Y"] == 0].iloc[:class_size,:], 
                     frame[frame["Y"] == 1].iloc[:class_size,:]], 
                    axis=0).sample(frac=1,random_state=state)
    
    X = np.array(sel.iloc[:,:512])
    y = list(sel["Y"])
    
    num_examples = len(X)
    i = 0
    inputs = []
    labels = []
    batches = []

    while i != num_examples:


        inputs.append(torch.tensor(X[i]))
        labels.append(torch.tensor(y[i]))

        if len(inputs) == bsize:
            batches.append([inputs,labels])
            inputs = []
            labels = []
        i+=1

    return batches

def arrange_val_batches(X_train, y_train, bsize):


    frame = pd.DataFrame(X_train)
    frame["Y"] = y_train
    frame = frame.sample(frac=1)
    class_size = min(Counter(y_train).values())
    
    sel = frame
    
    X = np.array(sel.iloc[:,:512])
    y = list(sel["Y"])
    num_examples = len(X)
    
    i = 0
    inputs = []
    labels = []
    batches = []

    while i != num_examples:


        inputs.append(torch.tensor(X[i]))
        labels.append(torch.tensor(y[i]))

        if len(inputs) == bsize:
            batches.append([inputs,labels])
            inputs = []
            labels = []
        i+=1

    return batches

   
X = pkl.load(open("/home/ext/konle/sentiment-datasets/bin_cls/train_data/ML_group/"+emotion+"/X.pkl","rb"))
Y = pkl.load(open("/home/ext/konle/sentiment-datasets/bin_cls/train_data/ML_group/"+emotion+"/Y.pkl","rb"))
S = pkl.load(open("/home/ext/konle/sentiment-datasets/bin_cls/train_data/ML_group/"+emotion+"/split.pkl","rb"))

bsize = args.batchsize
my_learning_rate = args.learningrate

indexes_train = np.where(S==0)[0]
indexes_test = np.where(S==1)[0]

X_train = X[indexes_train]
X_test = X[indexes_test]
y_train = Y[indexes_train]
y_test = Y[indexes_test]

batches = inner_shuffle_train(X_train, y_train, bsize, 43)

val_batches = arrange_val_batches(X_test, y_test, bsize)
optimizer = AdamW(model.parameters(),lr = my_learning_rate)
epochs = args.epochs
total_steps = len(batches) * epochs
scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps = 10, num_training_steps = total_steps, num_cycles=5)

device = "cuda"
model.to(device)

print("Train...")
report = ""
best_score = 0.3

for epoch_i in range(0, epochs):

    batches = inner_shuffle_train(X_train,y_train, bsize, epoch_i*10)
    print("Epoch: "+str(epoch_i))
    print(len(batches))
    total_loss = 0
    model.train()
    i = 0
    f1 = 0
    rep_t = []
    rep_p = []
    for step, batch in enumerate(batches):
        i+=1

        b_input_ids = torch.stack(batch[0]).to(device)
        b_labels = torch.stack(batch[1]).to(device)

        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None,labels=b_labels)

        loss = outputs[0].sum()
        total_loss += loss.item()


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        avg_train_loss = total_loss / i

        if i % 5 == 0:
            print("Batch Loss: "+str(i)+" "+str(avg_train_loss))
            report += "Batch Loss: "+str(i)+" "+str(avg_train_loss)+"\n"
            with open("/home/ext/konle/sentiment-datasets/bin_cls/ML_grouped/"+emotion+".txt", "w") as f:
                f.write(report)
    t = []
    p = []
    
    model.eval()
    
    with torch.no_grad():
        for batch in val_batches:
        
        
            
            b_input_ids = torch.stack(batch[0]).to(device)
            b_labels = torch.stack(batch[1]).to(device)


            outputs = model(b_input_ids, token_type_ids=None)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            tlabel = np.argmax(logits, axis=1)
            label_ids = b_labels.to('cpu').numpy()
            trues = label_ids.flatten()
            pred = tlabel.flatten()

            t.append(trues)
            p.append(pred)

    creport = classification_report(np.stack(t).flatten(), np.stack(p).flatten())
    print(creport)
    report+=creport
    f1_epoch = creport.split("\n")[-3]
    f1_epoch = float(re.sub("\s+"," ",f1_epoch).split(" ")[-2])
    
    if f1_epoch >= best_score:
        torch.save(model.state_dict(), "/home/ext/konle/sentiment-datasets/bin_cls/models/MLG_500_"+emotion+"_"+str(f1_epoch)+".pt")

    model.train()
