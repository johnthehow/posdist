import os
import re
import sys
import time
import datetime
import pickle
import torch
import shutil
from pathlib import Path
from torch import nn
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from collections import Counter

class dataset(Dataset):
	def __init__(self,datapairs):
		self.feats = [pair[:16] for pair in datapairs]
		self.targets = [int(pair[16].item()) for pair in datapairs]
	def __len__(self):
		return len(self.targets)
	def __getitem__(self,idx):
		feat = self.feats[idx]
		target = self.targets[idx]
		return feat,target

class network(nn.Module):
	def __init__(self):
		super().__init__()
		self.dense1 = nn.Linear(16,16)
		self.soft1 = nn.Softmax(dim=1)

	def forward(self,x):
		x = self.dense1(x)
		x = self.soft1(x)
		return x

def train(dataloader, model, loss_fn, optimizer,report_interval,train_rec_file):
    size = len(dataloader.dataset)
    model.train()
    num_batches = len(dataloader)
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % report_interval == 0:
            loss= loss.item()
            current = batch * len(X)
            train_rec_file.write(f"train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n")
    avg_loss = total_loss/num_batches
    return {'avg_loss':avg_loss}

def test(dataloader, model, loss_fn, train_rec_file):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    train_rec_file.write(f"test accuracy: {(100*correct):>0.8f}%, test avg loss: {test_loss:>8f} \n")
    return {'acc':correct,'loss':test_loss}

def train_onehead_probe(dataset_path,layer,head,split_ratio,train_batch_size,test_batch_size,report_interval,epochs,learn_rate):

	now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	model = network()
	layer = int(layer)
	head = int(head)
	loss_fn = CrossEntropyLoss()
	optimizer = SGD(model.parameters(),lr=learn_rate)
	print(f'Training layer-{layer:02d} head-{head:02d}...')

	with open(Path(dataset_path).joinpath(f'{layer:02d}_{head:02d}').joinpath('data').joinpath(f'attnrowlabs_{layer:02d}_{head:02d}.pkl'),mode='rb') as pkl:
		attnrowlabs = pickle.load(pkl)
	os.mkdir(Path(dataset_path).joinpath(f'{layer:02d}_{head:02d}').joinpath('res'))
	os.mkdir(Path(dataset_path).joinpath(f'{layer:02d}_{head:02d}').joinpath('env'))
	train_rec_file = open(Path(dataset_path).joinpath(f'{layer:02d}_{head:02d}').joinpath('res').joinpath(f'train_rec_{layer:02d}_{head:02d}_{now}.txt'),mode='w+')

	print(f'Layer {layer:02d} Head {head:02d}')
	
	
	poss = [int(i) for i in attnrowlabs[:,16].tolist()]
	pos_dist = Counter(poss)
	print(f'position distribution: {str(pos_dist)}')
	train_rec_file.write(f'position distribution: {str(pos_dist)}')
	train_rec_file.write('\n')
	baseline = pos_dist.most_common(1)[0][1]/len(poss)
	print(f'local majority baseline: {baseline}')
	train_rec_file.write(f'local majority baseline: {baseline}')
	train_rec_file.write('\n')


	split_point = int(len(poss)*split_ratio)
	train_datapairs = attnrowlabs[:len(poss)-split_point]
	test_datapairs = attnrowlabs[split_point:]
	train_dataset = dataset(train_datapairs)
	train_dataloader = DataLoader(train_dataset,batch_size=train_batch_size, shuffle=True)
	test_dataset = dataset(test_datapairs)
	test_dataloader = DataLoader(test_dataset,batch_size=test_batch_size, shuffle=True)
	train_rec_file.write(f'\n========Layer:{layer:02d} Head:{head:02d}===========\n')

	for e in range(epochs):
		train_rec_file.write(f"Epoch {e+1}\n-------------------------------\n")
		train_res = train(train_dataloader, model, loss_fn, optimizer, report_interval,train_rec_file)
		test_res = test(test_dataloader, model, loss_fn,train_rec_file)

	with open(dataset_path.joinpath(f'baseline_{baseline}'),mode='w+') as f:
		pass
	with open(dataset_path.joinpath('accuracies.txt'),mode='a+',encoding='utf-8') as f:
		f.write('layer\thead\tacc\n')
		f.write(f'{layer:02d}\t{head:02d}\t{test_res["acc"]:0.8f}\n')

	torch.save(model,Path(dataset_path).joinpath(f'{layer:02d}_{head:02d}').joinpath('res').joinpath(f'trained_mod_{layer:02d}_{head:02d}_acc_{test_res["acc"]:.6f}_{now}.pkl')) 

	train_rec_file.close()
	return

if __name__ == '__main__':
	split = float(sys.argv[1])
	batch = int(sys.argv[2])
	epochs = int(sys.argv[3])
	learn_rate = float(sys.argv[4])
	dataset_path = Path(sys.argv[5])
	for layer in range(1,13):
		for head in range(1,13):
			train_onehead_probe(dataset_path,layer,head,split,batch,batch,200,epochs,learn_rate)
	print(f'Train records and trained models saved at {dataset_path}')
	print('Done!')
