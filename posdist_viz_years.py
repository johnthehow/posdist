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

def corpus_loader(pkl_path):
	with open(pkl_path, mode='rb') as corpus_file:
		corpus_pkl = pickle.load(corpus_file)
	return corpus_pkl

def corpora_loader(yearlist,corpus_path):
	corpus_years = []
	for year in yearlist:
		new_corpus_path = re.sub('\d{4}',str(year),corpus_path)
		corpus_years.append(corpus_loader(new_corpus_path))
	return corpus_years


def posdist_word(word,corpus_pkl,sent_length):
	wordposlog = []
	lengthnsents = [i for i in corpus_pkl if len(i)==sent_length]
	print(f'posdist: Number of all sents of len {sent_length}: {len(lengthnsents)}')
	for sent in lengthnsents:
		if word in sent:
			wordposlog += [pos for pos,value in enumerate(sent) if value == word]
	wordposlog = [j+1 for j in wordposlog]
	fd = Counter(wordposlog)
	print(f'posdist: Sents containing word "{word}" in length {sent_length} sents: {len(wordposlog)} ')

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


def create_nested_dict(lists,idx): # 20230315092535
	lists = list(reversed(lists))
	def rec(lists,idx): # 20230315092535
		if idx>0:
			listpop = lists[idx]
			return {i:rec(lists,idx-1) for i in listpop}
		else:
			listpop = lists[idx]
			return {i:[] for i in listpop}
	return rec(lists, idx)

def draw_line_posdist_word_len_years_panels(rows,cols,sent_length,words,titles,corpora,year_selected,year_avail,img_fmt):
	fig = plt.figure(figsize=(5,15),dpi=300)
	axes = fig.subplots(rows,cols).flatten()
	word_cnt = 0
	image_data_container = create_nested_dict([[0,1,2],[0,1,2,3]], 1)
	for word in words: # 每种语言的一个单词
		markers = iter(['+', 'x', 'D', 's', 'o', '^', 'v'])
		linestyle_cnt = iter(range(7))
		year_cnt = 0
		for year in year_selected[word_cnt]: # year_selected: [[2005, 2006], [2005, 2006], [2007,2008]] year: 2005

			corpus_idx = year_avail[word_cnt].index(year)
			corpus = corpora[word_cnt][corpus_idx]
			xs = [i for i in range(1,sent_length+1)]
			ys = posdist_word(word,corpus,sent_length)['fd_density']
			xticks = [i for i in range(1,sent_length+1)]
			axes[word_cnt].set_xlabel('linear position in sentence')
			axes[word_cnt].set_ylabel('probability')
			axes[word_cnt].set_xticks(xticks)
			image_data = axes[word_cnt].plot(xs,ys,label=str(year), marker=next(markers), fillstyle='none', linewidth=1, linestyle=(0,(7,next(linestyle_cnt))))
			image_data_container[word_cnt][year_cnt] = image_data[0]			
			axes[word_cnt].legend(title='year')
			axes[word_cnt].set_title(titles[word_cnt])
			year_cnt += 1
		word_cnt += 1

	now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	plt.savefig(f'D:/{now}.{img_fmt}',format=img_fmt)
	return image_data_container

if __name__ == '__main__':
	# plt.rcParams["font.sans-serif"]=["SimHei"]
	# plt.rcParams["axes.unicode_minus"] = False
	sent_len = int(input('Sentence length: '))
	corpora_dir_en = Path(input('Path for English pickle corpora: '))
	corpora_dir_de = Path(input('Path for German pickle corpora: '))
	corpora_dir_es = Path(input('Path for Spanish pickle corpora: '))

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

	image_data_ctn = draw_line_posdist_word_len_years_panels(3,1,sent_len,words,titles,corpora,year_selected,year_avail,'png')

	now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	filename = Path('D:/').joinpath(now+'_imagedata.pkl')
	with open(filename, mode='wb') as file:
		pickle.dump(image_data_ctn, file)
	print(f'image raw data saved at {filename}')