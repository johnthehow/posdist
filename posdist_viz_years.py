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

def 文中直接使用的绘图函数():
	pass


# 函数ID: 2022012225031
# 注意: 本函数的前置代码必须使用, 是输入的参数
# 依赖函数: posdist_word
# 函数功能: 绘制同一单词同一长度不同年份的位频分布折线图 (3种语言) [文中使用]
# 前置代码↓ (内存使用63.3 GB) (已检查可用20220402145950)
# year_selected_en = [2005,2009,2013,2017]
# year_selected_de = [2007,2011,2015,2019]
# year_selected_es = [2008,2012,2016,2020]
# year_selected = [year_selected_en,year_selected_de,year_selected_es]
# year_avail_en = [2005,2006,2007,2008,2009,2010,2013,2014,2015,2016,2017,2018,2019,2020]
# year_avail_de = [2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]
# year_avail_es = [2006,2007,2008,2009,2010,2011,2012,2013,2016,2017,2018,2019,2020]
# year_avail = [year_avail_en,year_avail_de,year_avail_es]
# corpora_en = posdist_new.corpora_loader(year_avail_en,r'D:\thehow\3\POSDIST\CORPUS\2_CORPUS\EN\CODENRES\RES\YEAR\eng_news_2005_1M-sentences.pkl') # 这里没有错, 具体年份会被corpora_loader函数替换掉
# corpora_de = posdist_new.corpora_loader(year_avail_de,r'D:\thehow\3\POSDIST\CORPUS\2_CORPUS\DE\CODENRES\RES\YEAR\deu_news_2005_1M-sentences.pkl') # 这里没有错, 具体年份会被corpora_loader函数替换掉
# corpora_es = posdist_new.corpora_loader(year_avail_es,r'D:\thehow\3\POSDIST\CORPUS\2_CORPUS\ES\CODENRES\RES\YEAR\spa_news_2006_1M-sentences.pkl') # 这里没有错, 具体年份会被corpora_loader函数替换掉
# corpora = [corpora_en,corpora_de,corpora_es]
# words = ['but', 'auf', 'los']
# titles = ['but (English)', 'auf (German)', 'los (Spanish)']
# 示例用法: posdist_new.draw_line_posdist_word_len_years_panels(3,1,20,words,titles,corpora,year_selected,year_avail,'png')
def draw_line_posdist_word_len_years_panels(rows,cols,sent_length,words,titles,corpora,year_selected,year_avail,img_fmt):
	fig = plt.figure(figsize=(5,15),dpi=300)
	axes = fig.subplots(rows,cols).flatten()
	word_cnt = 0
	for word in words:
		for year in year_selected[word_cnt]:
			corpus_idx = year_avail[word_cnt].index(year)
			corpus = corpora[word_cnt][corpus_idx]
			xs = [i for i in range(1,sent_length+1)]
			ys = posdist_word(word,corpus,sent_length)['fd_density']
			xticks = [i for i in range(1,sent_length+1)]
			axes[word_cnt].set_xlabel('linear position in sentence')
			axes[word_cnt].set_ylabel('probability')
			axes[word_cnt].set_xticks(xticks)
			axes[word_cnt].plot(xs,ys,label=str(year))
			axes[word_cnt].legend(title='year')
			axes[word_cnt].set_title(titles[word_cnt])
		word_cnt += 1

	now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	plt.savefig(f'D:/{now}.{img_fmt}',format=img_fmt)
	return

if __name__ == '__main__':
	sent_len = int(input('Sentence length: '))
	corpora_dir_en = Path(input('Path for English corpora: '))
	corpora_dir_de = Path(input('Path for German corpora: '))
	corpora_dir_es = Path(input('Path for Spanish corpora: '))

	filelist_en = os.listdir(corpora_dir_en)
	filelist_de = os.listdir(corpora_dir_de)
	filelist_es = os.listdir(corpora_dir_es)

	yearlist_en = set()
	yearlist_de = set()
	yearlist_es = set()

	for fname in filelist_en:
		if fname.find('pkl') != -1:
			yearlist_en.add(int(re.search('\d{4}',fname)[0]))
	for fname in filelist_de:
		if fname.find('pkl') != -1:
			yearlist_de.add(int(re.search('\d{4}',fname)[0]))
	for fname in filelist_es:
		if fname.find('pkl') != -1:
			yearlist_es.add(int(re.search('\d{4}',fname)[0]))

	year_avail_en = sorted(yearlist_en)
	year_avail_de = sorted(yearlist_de)
	year_avail_es = sorted(yearlist_es)

	print('Years available for English')
	print(year_avail_en)
	print('Years available for German')
	print(year_avail_de)
	print('Years available for Spanish')
	print(year_avail_es)
	print('\n')

	year_avail = [year_avail_en,year_avail_de,year_avail_es]

	year_selected_en = input('Enter four years among available years of English (separated by space): ').split()
	year_selected_en = [int(i) for i in year_selected_en]
	year_selected_de = input('Enter four years among available years of German (separated by space): ').split()
	year_selected_de = [int(i) for i in year_selected_de]
	year_selected_es = input('Enter four years among available years of Spanish (separated by space): ').split()
	year_selected_es = [int(i) for i in year_selected_es]

	year_selected = [year_selected_en,year_selected_de,year_selected_es]

	word_en = input('A English function word: ')
	word_de = input('A German function word: ')
	word_es = input('A Spanish function word: ')

	words = [word_en, word_de, word_es]

	title_en = f'{word_en} (English)'
	title_de = f'{word_de} (German)'
	title_es = f'{word_es} (Spanish)'

	titles = [title_en,title_de,title_es]

	print('Loading corpora...')
	corpora_en = corpora_loader(year_avail_en,f'{corpora_dir_en.joinpath(filelist_en[0])}')
	corpora_de = corpora_loader(year_avail_de,f'{corpora_dir_de.joinpath(filelist_de[0])}')
	corpora_es = corpora_loader(year_avail_es,f'{corpora_dir_es.joinpath(filelist_es[0])}')
	corpora = [corpora_en,corpora_de,corpora_es]

	draw_line_posdist_word_len_years_panels(3,1,sent_len,words,titles,corpora,year_selected,year_avail,'png')
