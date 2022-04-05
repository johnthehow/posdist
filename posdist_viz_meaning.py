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


# 函数ID: 2022012242242
# 依赖函数: 无
# 参数解释:
#	wordlists: 手工编辑, 见前置代码
#	legends: 手工编辑, 间前置代码
# 	corpora_databases: 预计算的位频分布(density)
#		参数来源函数: 2022012030218
# 前置代码
# import pickle
# from thehow import posdist_new
# wordlist_and = ['and','und','et','y','и','a']
# wordlist_or = ['or', 'oder', 'ou','o','или','nebo']
# wordlist_i = ['i','ich','je','yo','я','já']
# wordlist_we = ['we','wir','nous','nosotros','мы','my']
# wordlists = [wordlist_and,wordlist_or,wordlist_i,wordlist_we]

# legend_and = ['and (English)','und (German)','et (French)','y (Spanish)','и (Russian)','a (Czech)']
# legend_or = ['or (English)', 'oder (German)','ou (French)','o (Spanish)','или (Russian)','nebo (Czech)']
# legend_i = ['i (English)','ich (German)','je (French)','yo (Spanish)','я (Russian)','já (Czech)']
# legend_we = ['we (English)','wir (German)','nous (French)','nosotros (Spanish)','мы (Russian)','my (Czech)']
# legends = [legend_and,legend_or,legend_i,legend_we]

# with open(r'D:\thehow\3\POSDIST\CORPUS\7_POSDIST_DENSITY\en_43words_4_37.pkl', mode='rb') as file:
# 	pdd_en = pickle.load(file)
# with open(r'D:\thehow\3\POSDIST\CORPUS\7_POSDIST_DENSITY\de_38words_4_37.pkl', mode='rb') as file:
# 	pdd_de = pickle.load(file)
# with open(r'D:\thehow\3\POSDIST\CORPUS\7_POSDIST_DENSITY\fr_41words_4_37.pkl', mode='rb') as file:
# 	pdd_fr = pickle.load(file)
# with open(r'D:\thehow\3\POSDIST\CORPUS\7_POSDIST_DENSITY\es_39words_4_37.pkl', mode='rb') as file:
# 	pdd_es = pickle.load(file)
# with open(r'D:\thehow\3\POSDIST\CORPUS\7_POSDIST_DENSITY\ru_39words_4_37.pkl', mode='rb') as file:
# 	pdd_ru = pickle.load(file)
# with open(r'D:\thehow\3\POSDIST\CORPUS\7_POSDIST_DENSITY\cz_37words_4_37.pkl', mode='rb') as file:
# 	pdd_cz = pickle.load(file)
# pdds = [pdd_en,pdd_de,pdd_fr,pdd_es,pdd_ru,pdd_cz]

# 示例用法: posdist_new.draw_line_posdist_meaning_langs_precalc_panels(wordlists,legends,16,pdds)
def draw_line_posdist_meaning_langs_precalc_panels(wordlists,legends,sent_length,corpora_databases, save_path):
	fig = plt.figure(figsize = (16,14),dpi=300)
	axes = fig.subplots(2,2).flatten()
	axes_cnt = 0
	for wordlist in wordlists:
		word_cnt = 0
		for word in wordlist:
			xs = [i+1 for i in range(sent_length)]
			ys = corpora_databases[word_cnt][sent_length][word]
			xticks = [i for i in range(1,sent_length+1)]
			axes[axes_cnt].plot(xs,ys)
			axes[axes_cnt].plot(xs,ys,label=legends[axes_cnt][word_cnt])
			axes[axes_cnt].legend(title='word')
			word_cnt += 1
		axes[axes_cnt].set_xticks(xticks)
		axes[axes_cnt].set_ylabel('probability')
		axes[axes_cnt].set_xlabel('linear position in sentence')
		axes_cnt += 1
	fig.subplots_adjust(wspace=0.15,hspace=0.15)
	# plt.show()
	now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	plt.savefig(f'{save_path.joinpath(now)}.png',format='png')
	print(f'Plot saved as {save_path.joinpath(now)}.png')
	return

if __name__ == '__main__':
	sent_len = int(input('Sentence length: '))
	word_group_1 = input('1st group of words of similar meaning, separated by spaces (EN, DE, FR, ES, RU, CZ): ').split()
	word_group_2 = input('2nd group of words of similar meaning, separated by spaces (EN, DE, FR, ES, RU, CZ): ').split()
	word_group_3 = input('3rd group of words of similar meaning, separated by spaces (EN, DE, FR, ES, RU, CZ): ').split()
	word_group_4 = input('4th group of words of similar meaning, separated by spaces (EN, DE, FR, ES, RU, CZ): ').split()
	word_groups = [word_group_1,word_group_2,word_group_3,word_group_4]

	legend_group_1 = [f'{word_group_1[0]} (English)', f'{word_group_1[1]} (German)', f'{word_group_1[2]} (French)', f'{word_group_1[3]} (Spanish)', f'{word_group_1[4]} (Russian)', f'{word_group_1[5]} (Czech)']
	legend_group_2 = [f'{word_group_2[0]} (English)', f'{word_group_2[1]} (German)', f'{word_group_2[2]} (French)', f'{word_group_2[3]} (Spanish)', f'{word_group_2[4]} (Russian)', f'{word_group_2[5]} (Czech)']
	legend_group_3 = [f'{word_group_3[0]} (English)', f'{word_group_3[1]} (German)', f'{word_group_3[2]} (French)', f'{word_group_3[3]} (Spanish)', f'{word_group_3[4]} (Russian)', f'{word_group_3[5]} (Czech)']
	legend_group_4 = [f'{word_group_4[0]} (English)', f'{word_group_4[1]} (German)', f'{word_group_4[2]} (French)', f'{word_group_4[3]} (Spanish)', f'{word_group_4[4]} (Russian)', f'{word_group_4[5]} (Czech)']
	legends = [legend_group_1,legend_group_2,legend_group_3,legend_group_4]

	corpus_path_en = Path(input('Path for English concatenated corpus: '))
	corpus_path_de = Path(input('Path for German concatenated corpus: '))
	corpus_path_fr = Path(input('Path for French concatenated corpus: '))
	corpus_path_es = Path(input('Path for Spanish concatenated corpus: '))
	corpus_path_ru = Path(input('Path for Russian concatenated corpus: '))
	corpus_path_cz = Path(input('Path for Czech concatenated corpus: '))

	save_path = Path(input('Path for result image: '))

	print('Loading corpora...')

	corpus_en = corpus_loader(corpus_path_en)
	corpus_de = corpus_loader(corpus_path_de)
	corpus_fr = corpus_loader(corpus_path_fr)
	corpus_es = corpus_loader(corpus_path_es)
	corpus_ru = corpus_loader(corpus_path_ru)
	corpus_cz = corpus_loader(corpus_path_cz)

	pdd_en = precalc_posdist_vwords_vlens_fd_density([word_group_1[0],word_group_2[0],word_group_3[0],word_group_4[0]],corpus_en,12,37)
	pdd_de = precalc_posdist_vwords_vlens_fd_density([word_group_1[1],word_group_2[1],word_group_3[1],word_group_4[1]],corpus_de,12,37)
	pdd_fr = precalc_posdist_vwords_vlens_fd_density([word_group_1[2],word_group_2[2],word_group_3[2],word_group_4[2]],corpus_fr,12,37)
	pdd_es = precalc_posdist_vwords_vlens_fd_density([word_group_1[3],word_group_2[3],word_group_3[3],word_group_4[3]],corpus_es,12,37)
	pdd_ru = precalc_posdist_vwords_vlens_fd_density([word_group_1[4],word_group_2[4],word_group_3[4],word_group_4[4]],corpus_ru,12,37)
	pdd_cz = precalc_posdist_vwords_vlens_fd_density([word_group_1[5],word_group_2[5],word_group_3[5],word_group_4[5]],corpus_cz,12,37)

	posdist_density_databases = [pdd_en, pdd_de, pdd_fr, pdd_es, pdd_ru, pdd_cz]

	draw_line_posdist_meaning_langs_precalc_panels(word_groups,legends,sent_len,posdist_density_databases,save_path)