'''
PosDist is a Tool for the research on in-sentence position-frequence distribution
'''
__author__ = 'John Thehow'

import warnings
import re
import os
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
	print(f'posdist: Number of all sents of len {sent_length}: {len(lengthnsents)}')
	#如果一句话中找到目标词, 把这个词的位置加入wordposlog
	for sent in lengthnsents:
		if word in sent:
			wordposlog += [pos for pos,value in enumerate(sent) if value == word]
	wordposlog = [j+1 for j in wordposlog]
	fd = Counter(wordposlog)
	print(f'posdist: Sents containing word "{word}" in length {sent_length} sents: {len(wordposlog)} ')
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

def 预计算函数():
	pass


# 函数ID: 2022012022550
# 依赖函数: 2022012030209
# 返回此列表中的所有词, 在句长区间内的所有句长的, 以概率表征的位频分布, 数据结构为字典
# 字典的第一级key为句长, 字典的第二级key为单词, 二级字典的value为list, len(list)==句长
def precalc_posdist_vwords_vlens_fd_count(wordlist,corpus,start_len,end_len):
	len_word_dict = dict()
	for length in range(start_len,end_len+1):
		len_dict = dict()
		for word in wordlist:
			len_dict[word] = posdist_word(word,corpus,length)['fd_count']
		len_word_dict[length] = len_dict
	return len_word_dict

# 函数ID: 2022012030218
# 依赖posdist
# 返回此列表中的所有词, 在句长区间内的所有句长的, 以概率表征的位频分布, 数据结构为字典
# 字典的第一级key为句长, 字典的第二级key为单词, 二级字典的value为list, len(list)==句长
def precalc_posdist_vwords_vlens_fd_density(wordlist,corpus,start_len,end_len):
	len_word_dict = dict()
	for length in range(start_len,end_len+1):
		len_dict = dict()
		for word in wordlist:
			len_dict[word] = posdist_word(word,corpus,length)['fd_density']
		len_word_dict[length] = len_dict
	return len_word_dict



# 函数ID: 2022012023649
# 依赖posdist
# 获得预计算log字典, 用于计算同词异长, 和同长异词的KL散度差异
# 计算结果非常巨大, 弃用
def precalc_posdist_vwords_vlens_log(wordlist,corpus,start_len,end_len):
	len_word_dict = dict()
	for length in range(start_len,end_len+1):
		len_dict = dict()
		for word in wordlist:
			len_dict[word] = posdist_word(word,corpus,length)['log']
		len_word_dict[length] = len_dict
	return len_word_dict


def 文中直接使用的绘图函数():
	pass

# 函数ID: 2022012070544
# 依赖函数: 无
# 依赖数据: LENOVO-PC: 20220402174129
# 函数功能: 绘制同一单词不同长度的位频分布折线图 (六种语言) [文中使用]
# 前置代码:
# with open(r'D:\thehow\3\POSDIST\CORPUS\7_POSDIST_DENSITY\en_43words_4_37.pkl',mode='rb') as pdd:
#	pdd_en = pickle.load(pdd)
# with open(r'D:\thehow\3\POSDIST\CORPUS\7_POSDIST_DENSITY\de_38words_4_37.pkl',mode='rb') as pdd:
#	pdd_de = pickle.load(pdd)
# with open(r'D:\thehow\3\POSDIST\CORPUS\7_POSDIST_DENSITY\fr_41words_4_37.pkl',mode='rb') as pdd:
#	pdd_fr = pickle.load(pdd)
# with open(r'D:\thehow\3\POSDIST\CORPUS\7_POSDIST_DENSITY\es_39words_4_37.pkl',mode='rb') as pdd:
#	pdd_es = pickle.load(pdd)
# with open(r'D:\thehow\3\POSDIST\CORPUS\7_POSDIST_DENSITY\ru_39words_4_37.pkl',mode='rb') as pdd:
#	pdd_ru = pickle.load(pdd)
# with open(r'D:\thehow\3\POSDIST\CORPUS\7_POSDIST_DENSITY\cz_37words_4_37.pkl',mode='rb') as pdd:
#	pdd_cz = pickle.load(pdd)
# pdds = [pdd_en,pdd_de,pdd_fr,pdd_es,pdd_ru,pdd_cz]
# words = ['of','als','avec','este','по','ale']
# titles = ['of (English)','als (German)','avec (French)','este (Spanish)','по (Russian)','ale (Czech)']
# 示例用法: draw_line_posdist_word_vlens_precalc_panels(3,2,words,titles,pdds)
def draw_line_posdist_word_vlens_precalc_panels(rows,cols,words,titles,posdist_density_databases):
	markerstyle = ['+','x','D','s','o','^','v']
	fig = plt.figure(figsize=(15,16),dpi=300)
	axes = fig.subplots(rows,cols).flatten()
	for i in range(rows*cols):
		style_cnt = 0
		for length in range(12,37,4):
			ys = posdist_density_databases[i][length][words[i]]
			xs = [j+1 for j in range(length)]
			axes[i].plot(xs,ys,label=str(length),marker=markerstyle[style_cnt],markersize=4,fillstyle='none',linewidth=1,linestyle=(0,(7,style_cnt)))
			style_cnt += 1
		axes[i].set_xticks([i for i in range(1,37)])
		axes[i].set_xticklabels([i for i in range(1,37)],rotation='vertical')
		axes[i].set_xlabel('linear position in sentence')
		axes[i].set_ylabel('probability')
		axes[i].set_title(titles[i])
		axes[i].legend(title = 'Sent Len')
	# plt.show()
	now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	plt.savefig(f'D:/{now}.png',format='png')
	return

if __name__ == '__main__':

	posdist_density_databases = precalc_posdist_vwords_vlens_fd_density(wordlist,corpus,12,37)
	draw_line_posdist_word_vlens_precalc_panels(rows,cols,words,titles,posdist_density_databases)