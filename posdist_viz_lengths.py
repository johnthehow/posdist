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

def draw_line_posdist_word_vlens_precalc_panels(rows,cols,words,titles,posdist_density_databases):
	markerstyle = ['+','x','D','s','o','^','v']
	fig = plt.figure(figsize=(15,16),dpi=300)
	axes = fig.subplots(rows,cols).flatten()
	image_data_container = create_nested_dict([[i for i in range(6)],[j for j in range(12,37,4)]], 1)
	for i in range(rows*cols):
		style_cnt = 0
		for length in range(12,37,4):
			ys = posdist_density_databases[i][length][words[i]]
			xs = [j+1 for j in range(length)]
			image_data = axes[i].plot(xs,ys,label=str(length),marker=markerstyle[style_cnt],markersize=4,fillstyle='none',linewidth=1,linestyle=(0,(7,style_cnt)))
			image_data_container[i][length] = image_data[0]
			style_cnt += 1
		axes[i].set_xticks([i for i in range(1,37)])
		axes[i].set_xticklabels([i for i in range(1,37)],rotation='vertical')
		axes[i].set_xlabel('linear position in sentence')
		axes[i].set_ylabel('probability')
		axes[i].set_title(titles[i])
		axes[i].legend(title = 'Sent Len')
	now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	plt.savefig(f'{now}.png',format='png')
	print(f'Plot saved as {now}.png')
	return image_data_container

if __name__ == '__main__':

	# word_en = input('English word: ')
	word_en = 'and'
	# word_de = input('German word: ')
	word_de = 'für'
	# word_fr = input('French word: ')
	word_fr = 'les'
	# word_es = input('Spanish word: ')
	word_es = 'pero'
	# word_ru = input('Russian word: ')
	word_ru = 'на'
	# word_cz = input('Czech word: ')
	word_cz = 'nebo'
	words = [word_en, word_de, word_fr, word_es, word_ru, word_cz]
	titles = [f'{word_en} (English)', f'{word_de} (German)', f'{word_fr} (French)', f'{word_es} (Spanish)', f'{word_ru} (Russian)', f'{word_cz} (Czech)']

	# corpus_path_en = Path(input('Path for English concatenated corpus: '))
	corpus_path_en = Path(r'D:\thehow\TERMS\3\POSDIST\CORPUS\2_CORPUS\EN\CODENRES\RES\CAT\en_lepzig_all.pkl')
	# corpus_path_de = Path(input('Path for German concatenated corpus: '))
	corpus_path_de = Path(r'D:\thehow\TERMS\3\POSDIST\CORPUS\2_CORPUS\DE\CODENRES\RES\CAT\de_lepzig_all.pkl')
	# corpus_path_fr = Path(input('Path for French concatenated corpus: '))
	corpus_path_fr = Path(r'D:\thehow\TERMS\3\POSDIST\CORPUS\2_CORPUS\FR\CODENRES\RES\CAT\fr_lepzig_all.pkl')
	# corpus_path_es = Path(input('Path for Spanish concatenated corpus: '))
	corpus_path_es = Path(r'D:\thehow\TERMS\3\POSDIST\CORPUS\2_CORPUS\ES\CODENRES\RES\CAT\es_lepzig_all.pkl')
	# corpus_path_ru = Path(input('Path for Russian concatenated corpus: '))
	corpus_path_ru = Path(r'D:\thehow\TERMS\3\POSDIST\CORPUS\2_CORPUS\RU\CODENRES\RES\CAT\ru_lepzig_all.pkl')
	# corpus_path_cz = Path(input('Path for Czech concatenated corpus: '))
	corpus_path_cz = Path(r'D:\thehow\TERMS\3\POSDIST\CORPUS\2_CORPUS\CZ\CODENRES\RES\CAT\cz_lepzig_all.pkl')
	# save_path = Path(input('Path for result image: '))

	corpus_en = corpus_loader(corpus_path_en)
	corpus_de = corpus_loader(corpus_path_de)
	corpus_fr = corpus_loader(corpus_path_fr)
	corpus_es = corpus_loader(corpus_path_es)
	corpus_ru = corpus_loader(corpus_path_ru)
	corpus_cz = corpus_loader(corpus_path_cz)

	pdd_en = precalc_posdist_vwords_vlens_fd_density([word_en],corpus_en,12,37)
	pdd_de = precalc_posdist_vwords_vlens_fd_density([word_de],corpus_de,12,37)
	pdd_fr = precalc_posdist_vwords_vlens_fd_density([word_fr],corpus_fr,12,37)
	pdd_es = precalc_posdist_vwords_vlens_fd_density([word_es],corpus_es,12,37)
	pdd_ru = precalc_posdist_vwords_vlens_fd_density([word_ru],corpus_ru,12,37)
	pdd_cz = precalc_posdist_vwords_vlens_fd_density([word_cz],corpus_cz,12,37)

	posdist_density_databases = [pdd_en, pdd_de, pdd_fr, pdd_es, pdd_ru, pdd_cz]
	res = draw_line_posdist_word_vlens_precalc_panels(3,2,words,titles,posdist_density_databases,save_path)

	with open('posdist_viz_lengths_image_data.pkl', mode='wb') as file:
		pickle.dump(res, file)
	print(f'lengths image data saved at same directory')