'''
[模块注释]
	[功能]
		1. 主要功能: 生成平均注意力分布可视化(中文论文中图6)
	[用例]
		python attndis_viz.py 注意力行标签数据库目录 可视化结果保存目录 目标词 注意力层号 注意力头号
	[输入]
		注意力行标签数据库目录结构必须为:
			注意力行标签数据库目录
				句长1
					01_01
						data
							attnrowlabs_01_01.pkl
					...
					12_12
						data
							attnrowlabs_12_12.pkl
				...
				句长n
					01_01
						data
							attnrowlabs_01_01.pkl
					...
					12_12
						data
							attnrowlabs_12_12.pkl
	[依赖]
		1. 输入依赖
			1.1 高度依赖attn_rowpos.py的输出, 文件名高度敏感
'''

import pickle
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import argparse


'''
[函数注释]
	[功能]
		1. 主要功能: 生成平均注意力分布可视化
		2. 额外功能
	[设计图]
		1. 索引码: 
		2. 文件类型: 
	[参数]
		1. word
			1. 数据类型: string
			2. 数据结构: string
			3. 参数类型: 必选
			4. 语义: 目标词
			5. 取值范围: 
			6. 获得来源: 手动输入
			7. 样例文件/输入: 'and'
		2. lay
			1. 数据类型: int
			2. 数据结构: int
			3. 参数类型: 必选
			4. 语义: 注意力层号
			5. 取值范围: [1,12]
			6. 获得来源: 手动输入
			7. 样例文件/输入: 3
		3. head
			1. 数据类型: int
			2. 数据结构: int
			3. 参数类型: 必选
			4. 语义: 注意力头号
			5. 取值范围: [1,12]
			6. 获得来源: 手动输入
			7. 样例文件/输入: 10
		4. picklepath
			1. 数据类型: string
			2. 数据结构: string
			3. 参数类型: 必选
			4. 语义: 作为注意力行+位置标签的pickle路径
			5. 取值范围: 
			6. 获得来源: 手动输入, attn_rowpos.py的输出
			7. 样例文件/输入: 'C:\ATTNROWLABS', 20230810185741.zip
				7.1 注意力行标签数据库目录结构必须为:
					C:\ATTNROWLABS
						12
							01_01
								data
									attnrowlabs_01_01.pkl
							...
							12_12
								data
									attnrowlabs_12_12.pkl
						...
						36
							01_01
								data
									attnrowlabs_01_01.pkl
							...
							12_12
								data
									attnrowlabs_12_12.pkl
				7.2 attnrowlabs_xx_xx.pkl文件的数据结构
					7.2.1 数据类型: torch.tensor
					7.2.2 形状: 句子数, 句长+1
						7.2.2.1 例如, 对应句长24, 输入为24000句时, 尺寸为 24000,25, 最后一列为整数位置标签
		5. savepath
			1. 数据类型: string
			2. 数据结构: string
			3. 参数类型: 必选
			4. 语义: 保存可视化结果的路径
			5. 取值范围: 
			6. 获得来源: 手动输入
			7. 样例文件/输入: 'C:\VIZ'

	[用例]
		posprobe_viz('and',3,1,'C:\ATTNROWLABS','C:\VIZ')
			1. 输出
				1. 语义: 目标词'and'在3-1注意力头的平均注意力分布可视化
				2. 数据类型: 无
				3. 数据结构: 无
				4. 样例文件/输出: 20230810185242.png 
	[依赖]
		1. 
		2. 
	[已知问题]
		1. [问题1标题]
			1. 问题描述
			2. 问题复现
				1. 复现环境
				2. 复现语句
				3. 复现存档
	[开发计划]
		1. 
		2.
	[备注]
		1. 只支持生成7个句长的图片,多了不行, 少了可以
		2. 输入必须严格来自attn_rowpos.py
		3. 默认输出句长'12','16','20','24','28','32','36'的可视化, 不可更改
'''
def posprobe_viz(word,lay,head,picklepath,savepath):
	lens = ['12','16','20','24','28','32','36']
	pklsubpaths = [f'{slen}/{lay:02d}_{head:02d}/data/attnrowlabs_{lay:02d}_{head:02d}.pkl' for slen in lens]
	pklfulpaths = [picklepath.joinpath(i) for i in pklsubpaths]

	pkls = []
	for p in pklfulpaths:
		with open(p,mode='rb') as file:
			pkls.append(pickle.load(file))

	pkmeans = []
	for pk in pkls:
		pk_trim = pk[:,:-1]
		pk_trim_mean = pk_trim.mean(axis=0)
		pk_trim_mean_zero = torch.cat((pk_trim_mean,torch.tensor(0).reshape(1)))
		pkmeans.append(pk_trim_mean_zero)

	fig = plt.figure(dpi=400)
	ax = fig.subplots()
	markers = iter(['+', 'x', 'D', 's', 'o', '^', 'v'])
	linestyle_cnt = iter(range(7))
	for pkm in pkmeans:
		xlen = len(pkm)-1
		ax.plot([i for i in range(1,xlen+2)], pkm, label=str(xlen), marker=next(markers), fillstyle='none', linewidth=1, linestyle=(0,(7,next(linestyle_cnt))))
	ax.set_xticks([i for i in range(1,37)])
	ax.set_xticklabels([i for i in range(1,37)],rotation='vertical')
	ax.set_xlabel('句中线性位置')
	ax.set_ylabel('平均注意力值')
	ax.legend(title = '句长')
	plt.title(f'功能词: {word} (英语), 注意力头 {lay:02d}-{head:02d}')

	save_filename = savepath.joinpath(f'{word}_{lay:02d}_{head:02d}_12_36.png')
	plt.savefig(save_filename, format='png')
	plt.cla()
	plt.clf()
	plt.close()
	return

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('pklpath')
	parser.add_argument('vizpath')
	parser.add_argument('word')
	parser.add_argument('layer')
	parser.add_argument('head')
	args = parser.parse_args()

	plt.rcParams["font.sans-serif"]=["SimHei"]
	plt.rcParams["axes.unicode_minus"]=False
	posprobe_viz(args.word, int(args.layer), int(args.head), Path(args.pklpath), Path(args.vizpath))
