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

def sent_selector(word,sent_length,corpus_pkl):
	sents_len = [sent for sent in corpus_pkl if len(sent) == sent_length]
	sents_len_word = [' '.join(sent) for sent in sents_len if word in sent]
	return sents_len_word

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

def posdist_phrase(corpus_pkl,sent_length,phrase):
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

def precalc_posdist_vwords_vlens_fd_count(wordlist,corpus,start_len,end_len):
	len_word_dict = dict()
	for length in range(start_len,end_len+1):
		len_dict = dict()
		for word in wordlist:
			len_dict[word] = posdist_word(word,corpus,length)['fd_count']
		len_word_dict[length] = len_dict
	return len_word_dict

def precalc_posdist_vwords_vlens_fd_density(wordlist,corpus,start_len,end_len):
	len_word_dict = dict()
	for length in range(start_len,end_len+1):
		len_dict = dict()
		for word in wordlist:
			len_dict[word] = posdist_word(word,corpus,length)['fd_density']
		len_word_dict[length] = len_dict
	return len_word_dict

def precalc_posdist_vwords_vlens_log(wordlist,corpus,start_len,end_len):
	len_word_dict = dict()
	for length in range(start_len,end_len+1):
		len_dict = dict()
		for word in wordlist:
			len_dict[word] = posdist_word(word,corpus,length)['log']
		len_word_dict[length] = len_dict
	return len_word_dict

def draw_line_posdist_meaning_langs_precalc_panels(wordlists,legends,sent_length,corpora_databases, save_path):
	fig = plt.figure(figsize = (16,14),dpi=300)
	axes = fig.subplots(2,2).flatten()
	axes_cnt = 0
	for wordlist in wordlists: # wordlists: [[and, und, et], [or, oder, ou], ...]
		markers = iter(['+', 'x', 'D', 's', 'o', '^', 'v'])
		linestyle_cnt = iter(range(7))
		word_cnt = 0
		for word in wordlist:
			xs = [i+1 for i in range(sent_length)]
			ys = corpora_databases[word_cnt][sent_length][word]
			xticks = [i for i in range(1,sent_length+1)]
			# axes[axes_cnt].plot(xs,ys)
			axes[axes_cnt].plot(xs,ys,label=legends[axes_cnt][word_cnt],marker=next(markers), fillstyle='none', linewidth=1, linestyle=(0,(7,next(linestyle_cnt))))
			axes[axes_cnt].legend(title='word')
			word_cnt += 1
		axes[axes_cnt].set_xticks(xticks)
		axes[axes_cnt].set_ylabel('probability')
		axes[axes_cnt].set_xlabel('linear position in sentence')
		axes_cnt += 1
	fig.subplots_adjust(wspace=0.15,hspace=0.15)
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

	now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	filename = save_path.joinpath(now_time+'.pkl')
	with open(filename, mode='wb') as file:
		pickle.dump(posdist_density_databases, file)
	print(f'data file save at {filename}')