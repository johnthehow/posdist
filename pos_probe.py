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

# 定义数据集结构
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

# 搭建probe
class network(nn.Module):
	def __init__(self):
		super().__init__()
		self.dense1 = nn.Linear(16,16)
		self.soft1 = nn.Softmax(dim=1)

	def forward(self,x):
		x = self.dense1(x)
		x = self.soft1(x)
		return x

# 定义训练步骤
def train(dataloader, model, loss_fn, optimizer,report_interval,train_rec_file):
    size = len(dataloader.dataset)
    model.train() # 将模型设置为训练模式
    num_batches = len(dataloader)
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad() # 清零参数tensor中保存的gradient [DJDI]
        loss.backward() # 求所有支持gradient的tensor的梯度值
        optimizer.step()

        # 汇报训练情况
        if batch % report_interval == 0:
            loss= loss.item()
            current = batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            train_rec_file.write(f"train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n")
    avg_loss = total_loss/num_batches
    return {'avg_loss':avg_loss}

# 定义测试步骤
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
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    train_rec_file.write(f"test accuracy: {(100*correct):>0.8f}%, test avg loss: {test_loss:>8f} \n")
    return {'acc':correct,'loss':test_loss}

# 函数功能: 主函数: 训练逻辑
# 函数ID: 20220403180913
# 依赖函数:
# 	network
# 	
# 参数解释:
# 	dataset_path: 按 dataset_path/00_00/data/attnrowlabs_00_00.pkl 形式存放的训练数据目录, 同时也是结果保存目录
# 	layer: 从1开始计数的attention层号
# 	head: 从1开始计数的head号
# 	split_ratio: 训练的train/test比值
# 	train_batch_size: 训练batch size
# 	test_batch_size: 测试batch size
# 	report_interval: 多少个batch记录一条训练报告
# 	epochs: 训练多少个epoch
# 	learn_rate: 学习率
def train_onehead_probe(dataset_path,layer,head,split_ratio,train_batch_size,test_batch_size,report_interval,epochs,learn_rate):

	# 训练模型配置
	now = datetime.datetime.now().strftime('%Y%m%d%H%M%S') # 当前时间字符串
	model = network() # 初始化/实例化模型
	layer = int(layer)
	head = int(head)
	loss_fn = CrossEntropyLoss()
	optimizer = SGD(model.parameters(),lr=learn_rate)
	print(f'Training layer-{layer:02d} head-{head:02d}...')

	# 载入数据文件
	with open(Path(dataset_path).joinpath(f'{layer:02d}_{head:02d}').joinpath('data').joinpath(f'attnrowlabs_{layer:02d}_{head:02d}.pkl'),mode='rb') as pkl:
		attnrowlabs = pickle.load(pkl)
	# 创建训练记录目录和文件
	os.mkdir(Path(dataset_path).joinpath(f'{layer:02d}_{head:02d}').joinpath('res')) # 运行结果保存目录
	os.mkdir(Path(dataset_path).joinpath(f'{layer:02d}_{head:02d}').joinpath('env')) # 运行环境保存路径
	train_rec_file = open(Path(dataset_path).joinpath(f'{layer:02d}_{head:02d}').joinpath('res').joinpath(f'train_rec_{layer:02d}_{head:02d}_{now}.txt'),mode='w+') # 创建训练记录文件

	# 开始训练
	print(f'Layer {layer:02d} Head {head:02d}')
	
	
	# baseline打印
	poss = [int(i) for i in attnrowlabs[:,16].tolist()]
	pos_dist = Counter(poss) # 标签分布
	print(f'position distribution: {str(pos_dist)}')
	train_rec_file.write(f'position distribution: {str(pos_dist)}')
	train_rec_file.write('\n')
	baseline = pos_dist.most_common(1)[0][1]/len(poss) # most_common baseline
	print(f'local majority baseline: {baseline}')
	train_rec_file.write(f'local majority baseline: {baseline}')
	train_rec_file.write('\n')


	# 数据拆分并载入dataloader
	split_point = int(len(poss)*split_ratio)
	train_datapairs = attnrowlabs[:len(poss)-split_point]
	test_datapairs = attnrowlabs[split_point:]
	train_dataset = dataset(train_datapairs)
	train_dataloader = DataLoader(train_dataset,batch_size=train_batch_size, shuffle=True)
	test_dataset = dataset(test_datapairs)
	test_dataloader = DataLoader(test_dataset,batch_size=test_batch_size, shuffle=True)
	train_rec_file.write(f'\n========Layer:{layer:02d} Head:{head:02d}===========\n')

	# 正式开始训练
	for e in range(epochs):
		# print(f"Epoch {e+1}\n-------------------------------")
		train_rec_file.write(f"Epoch {e+1}\n-------------------------------\n")
		train_res = train(train_dataloader, model, loss_fn, optimizer, report_interval,train_rec_file)
		test_res = test(test_dataloader, model, loss_fn,train_rec_file)

	with open(dataset_path.joinpath(f'baseline_{baseline}'),mode='w+') as f:
		pass
	# 保存该attention head的精度
	with open(dataset_path.joinpath('accuracies.txt'),mode='a+',encoding='utf-8') as f:
		f.write('layer\thead\tacc\n')
		f.write(f'{layer}\t{head}\t{test_res[acc]:>0.8f}\n')


	# 保存已训练模型路径
	torch.save(model,Path(dataset_path).joinpath(f'{layer:02d}_{head:02d}').joinpath('res').joinpath(f'trained_mod_{layer:02d}_{head:02d}_acc_{test_res["acc"]:.6f}_{now}.pkl')) 

	# 收尾
	train_rec_file.close()
	return

# IDEL运行版代码
# dataset_path = Path(input('Heads directory: '))
# layer = input('Layer: ')
# head = input('Head: ')
# epochs = int(input('Epochs: '))




# 执行函数(一次性训练一个单词的所有head)


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
