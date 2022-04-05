'''
PosDist is a Tool for the research on in-sentence position-frequence distribution
'''
__author__ = 'John Thehow'

import warnings
import re
import os
import sys
import pickle
import random
import numpy
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from scipy import stats
from scipy.special import kl_div
import datetime

warnings.filterwarnings("ignore")
def 周边服务函数():
	pass

# 函数ID: 2022012025946
# 依赖函数: 无
# 函数功能: 载入语料库
def corpus_loader(pkl_path):
	'''载入PKL格式的清理过的tokenized_lines形式的语料库'''
	with open(pkl_path, mode='rb') as corpus_file:
		corpus_pkl = pickle.load(corpus_file)
	return corpus_pkl

# 函数ID: 2022012062530
# 依赖函数: 2022012025946
# 参数解释: corpus_path是一年pkl的文件路径, 函数会自动根据yearlist替换年份
def corpora_loader(yearlist,corpus_path):
	corpus_years = []
	for year in yearlist:
		new_corpus_path = re.sub('\d{4}',str(year),corpus_path)
		corpus_years.append(corpus_loader(new_corpus_path))
	return corpus_years

# 函数ID: 2022012062559
# 依赖函数: 无
# 函数功能: 从语料库PKL对象(tokenized_lines)中, 提取 包含目标词的 指定句长的 tokenized_lines
def sent_selector(word,sent_length,corpus_pkl):
	'''从载入的语料库中选择包含*目标词的*指定长度的句子'''
	sents_len = [sent for sent in corpus_pkl if len(sent) == sent_length]
	sents_len_word = [' '.join(sent) for sent in sents_len if word in sent]
	return sents_len_word

def 位频分布基本统计函数():
	pass

# 函数ID: 2022012030209
# 依赖: 无
# 功能: 从语料库PKL对象中, 获得目标词 在 指定句长的 位频分布
def posdist_word(word,corpus_pkl,sent_length):
	'''寻找目标长度句子中的word of interest的所有出现, 并记录该词的位置编号, 返回编号分布和原始编号记录'''
	# 目标词位置编号容器
	wordposlog = []
	#挑出句长为n的句子
	lengthnsents = [i for i in corpus_pkl if len(i)==sent_length]
	print(f'No. sentences of length-{sent_length}: {len(lengthnsents)}')
	#如果一句话中找到目标词, 把这个词的位置加入wordposlog
	for sent in lengthnsents:
		if word in sent:
			wordposlog += [pos for pos,value in enumerate(sent) if value == word]
	wordposlog = [j+1 for j in wordposlog]
	fd = Counter(wordposlog)
	print(f'No. of sentences containing word "{word}" in length-{sent_length} sentences: {len(wordposlog)} ')
	# 返回词在指定句长的位频分布(以概率表示), 无occurrence的位置概率为0
	fd_sorted = sorted(fd.items(), key = lambda kv: kv[0])
	fd_density = [v/len(wordposlog) for k,v in fd_sorted]
	fd_count = [v for k,v in fd_sorted]
	if len(fd_density) != sent_length:
		diff = list(set([i+1 for i in range(sent_length)])-set([pair[0] for pair in fd_sorted]))
		print(f'zero occurrence of word "{word}" on {diff}')
		for i in diff:
			fd_density.insert(i-1,0)
		print(fd_density)
	if len(fd_count) != sent_length:
		diff = list(set([i+1 for i in range(sent_length)])-set([pair[0] for pair in fd_sorted]))
		print(f'zero occurrence of word "{word}" on {diff}')
		for i in diff:
			fd_count.insert(i-1,0)
		print(fd_count)
	return {'fd':fd,'log':wordposlog,'fd_density':fd_density,'fd_count':fd_count}

# 函数ID: 2022012030558
# 依赖 无
# 计算短语的位频分布
def posdist_phrase(corpus_pkl,sent_length,phrase):
	'''仅适用于两词短语, phrase是空格分隔的一个字符串'''
	phraseposlog = []
	lengthnsents = [i for i in corpus_pkl if len(i)==sent_length]
	for sent in lengthnsents:
		if phrase.split()[0] in sent and phrase.split()[1] in sent:
			word1pos = [pos for pos,value in enumerate(sent) if value == phrase.split()[0]]
			word2pos = [pos for pos,value in enumerate(sent) if value == phrase.split()[1]]
			for word1idx in word1pos:
				if word1idx+1 in word2pos:
					phraseposlog.append(word1idx)
	phraseposlog = [j+1 for j in phraseposlog]
	fd = Counter(phraseposlog)
	print(f'posdist_phrase: {len(phraseposlog)} phrases matched in length {sent_length} sents')
	return {'fd':fd,'log':phraseposlog}

# 函数ID: 2022012062617
# 依赖函数: 2022012030209
# 多个单词当做一个单词, 统计其在指定句长的位频分布
def posdist_multiword(corpus_pkl,sent_length,*words):
	wordposlog = []
	lengthnsents = [i for i in corpus_pkl if len(i)==sent_length]
	for sent in lengthnsents:
		for word in words:
			if word in sent:
				wordposlog += [pos for pos,value in enumerate(sent) if value == word]
	wordposlog = [j+1 for j in wordposlog]
	fd = Counter(wordposlog)
	print(f'posdist_multiword: {len(wordposlog)} words matched in length {sent_length} sents')
	return {'fd':fd,'log':wordposlog}


def 量化差异对比函数():
	pass


# 函数ID: 2022012025739
# 依赖函数: 2022012030209
# 同一单词, 不同句长 的 KDE平滑的 对齐到第一个句长的y轴上限的 位频分布曲线们 之间的 两两KL散度差异
def diff_word_vlens(word,corpus_pkl,start_len,end_len,bandwidth):
	'''bandwidth控制KDE曲线的平滑程度, 越大越平滑, 但是越失真, 一般取1\ninterval决定隔多少句长间隔画一次线'''
	# 一个单词的KDE分布: 获得一个单词在不同句长的拉伸后KDE分布(density), 对齐到到第一个句长的高度
	# 获得第一个句长的高度, 用于后边句长的对齐
	first_log = posdist_word(word,corpus_pkl,start_len)['log']
	first_kde = stats.gaussian_kde(first_log,bw_method=bandwidth)
	first_xs = numpy.linspace(1,start_len,1000)
	first_ys = first_kde(first_xs)
	first_ys_max = first_ys.max()
	
	yss = []
	# 对于每个句长, 计算
	for sentlen in range(start_len,end_len+1,1):
		# 获得数据
		log = posdist_word(word,corpus_pkl,sentlen)['log']
		# 作图
		kde = stats.gaussian_kde(log,bw_method=bandwidth)
		xs = numpy.linspace(1,sentlen,1000)
		ys = kde(xs)
		ys = ys * (first_ys_max/ys.max()) #对齐到第一个句长的高度
		yss.append(ys)

	# 获得两两对比的项
	pairs = []
	for i in range(len(yss)):
		for j in range(len(yss)):
			if i<j:
				pairs.append((i,j))

	# 两两对比后的KLD差异均值和标准差
	klds = []
	for pair in pairs:
		kld = kl_div(yss[pair[0]],yss[pair[1]])
		klds.append(kld.sum())
	kldmean = sum(klds)/len(klds)
	kldstd = numpy.std(klds)
	return kldmean,kldstd



# 函数ID: 2022012060613
# 依赖函数: 2022012030209
# 同一句长, 不同单词 的 KDE平滑 的 位频分布曲线们 之间的 两两KL散度差异
def diff_len_vwords(wordlist,corpus_pkl,sent_length,start_len,end_len,bandwidth):
	yss = []
	for word in wordlist:
		log = posdist_word(word,corpus_pkl,sent_length)['log']
		kde = stats.gaussian_kde(log,bw_method=bandwidth)
		xs = numpy.linspace(1,sent_length,1000)
		ys = kde(xs)
		yss.append(ys)
	pairs = []
	for i in range(len(yss)):
		for j in range(len(yss)):
			if i<j:
				pairs.append((i,j))
	klds = []
	for p in pairs:
		kld = kl_div(yss[p[0]],yss[p[1]])
		klds.append(kld.sum())
	kldmean = sum(klds)/len(klds)
	kldstd = numpy.std(klds)
	print(kldmean)
	print(kldstd)
	return kldmean,kldstd





# 函数ID: 2022012061750
# 依赖函数: 2022012030209
# 函数功能: 同一单词, 同一句长, 不同年份之间的 位频分布KDE曲线 之间的 KL散度 差异
def diff_word_len_vyears(word,sent_length,corpora,bandwidth):
	yss = []
	for corpus in corpora:
		log = posdist_word(word,corpus,sent_length)['log']
		kde = stats.gaussian_kde(log,bw_method=bandwidth)
		xs = numpy.linspace(1,sent_length,1000)
		ys = kde(xs)
		yss.append(ys)
	pairs = []
	for i in range(len(yss)):
		for j in range(len(yss)):
			if i<j:
				pairs.append((i,j))
	klds = []
	for p in pairs:
		kld = kl_div(yss[p[0]],yss[p[1]])
		klds.append(kld.sum())
	kldmean = sum(klds)/len(klds)
	kldstd = numpy.std(klds)
	return kldmean,kldstd



def 文中使用的量化差异对比函数():
	pass


# 函数ID: 2022012061510
# 依赖函数: 2022012025739
# 同一单词, 不同句长 的 KDE平滑的 对齐到第一个句长的y轴上限的 位频分布曲线们 之间的 两两KL散度差异
def diff_vwords_vlens(wordlist,corpus_pkl,start_len,end_len,bandwidth):
	word_kl_means = []
	word_kl_stds = []
	for word in wordlist:
		print(f'{word}')
		meanstd = diff_word_vlens(word,corpus_pkl,start_len,end_len,bandwidth)
		word_kl_means.append(meanstd[0])
		word_kl_stds.append(meanstd[1])
		print(meanstd[0])
		print(meanstd[1])
	word_kl_means_mean = numpy.mean(word_kl_means)
	word_kl_stds_mean = numpy.std(word_kl_stds)
	return word_kl_means_mean, word_kl_stds_mean

# 函数ID: 2022012060738
# 依赖函数: 2022012060613
# 同一句长, 不同单词 的 KDE平滑的 位频分布曲线们 之间的 两两KL散度差异 (扩展到所有句长)
def diff_vlens_vwords(wordlist,corpus_pkl,start_len,end_len,bandwidth):
	kl_means = []
	kl_stds = []
	for i in range(start_len,end_len+1):
		meanstd = diff_len_vwords(wordlist,corpus_pkl,i,start_len,end_len,bandwidth)
		kl_means.append(meanstd[0])
		kl_stds.append(meanstd[1])
	kl_mean_means = numpy.mean(kl_means)
	kl_mean_stds = numpy.mean(kl_stds)
	return kl_mean_means,kl_mean_stds

# 函数ID: 2022012062321
# 依赖函数: 2022012061750
# 函数功能: 同一单词, 同一句长, 不同年份之间的 位频分布KDE曲线 之间的 KL散度 差异 (扩展至各单词, 各句长)
def diff_vwords_vlens_vyears(wordlist,start_len,end_len,corpora,bandwidth):
	kldmeans = []
	kldstds = []
	for word in wordlist:
		for length in range(start_len,end_len+1):
			kldmeanstd = diff_word_len_vyears(word,length,corpora,bandwidth)
			kldmeans.append(kldmeanstd[0])
			kldstds.append(kldmeanstd[1])
	kldmeans_avg = numpy.mean(kldmeans)
	kldstds_avg = numpy.mean(kldstds)
	print(f'Mean: {kldmeans_avg}')
	print(f'SD: {kldstds_avg}')
	return kldmeans_avg,kldstds_avg

if __name__ == '__main__':
	start_len = int(sys.argv[1])
	end_len = int(sys.argv[2])
	bandwidth = float(sys.argv[3])
	corpora_dir = Path(sys.argv[4])
	wordlist = sys.argv[5].split()
	yearlist = set()
	for fname in os.listdir(corpora_dir):
		if fname.find('pkl') != -1:
			yearlist.add(re.search('\d{4}',fname)[0])
	yearlist = sorted(yearlist)
	print(yearlist)
	corpora = corpora_loader(yearlist,corpora_dir)
	res = diff_vwords_vlens_vyears(wordlist,start_len,end_len,corpus)
