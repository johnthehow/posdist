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

def corpus_loader(pkl_path):
	print('Loading corpus...')
	with open(pkl_path, mode='rb') as corpus_file:
		corpus_pkl = pickle.load(corpus_file)
	return corpus_pkl

def corpora_loader(yearlist,corpus_path):
	print('Loading corpora...')
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
	print(f'No. sentences of length-{sent_length}: {len(lengthnsents)}')
	for sent in lengthnsents:
		if word in sent:
			wordposlog += [pos for pos,value in enumerate(sent) if value == word]
	wordposlog = [j+1 for j in wordposlog]
	fd = Counter(wordposlog)
	print(f'No. of sentences containing word "{word}" in length-{sent_length} sentences: {len(wordposlog)} ')
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


def diff_word_vlens(word,corpus_pkl,start_len,end_len,bandwidth):
	first_log = posdist_word(word,corpus_pkl,start_len)['log']
	first_kde = stats.gaussian_kde(first_log,bw_method=bandwidth)
	first_xs = numpy.linspace(1,start_len,1000)
	first_ys = first_kde(first_xs)
	first_ys_max = first_ys.max()
	
	yss = []
	for sentlen in range(start_len,end_len+1,1):
		log = posdist_word(word,corpus_pkl,sentlen)['log']
		kde = stats.gaussian_kde(log,bw_method=bandwidth)
		xs = numpy.linspace(1,sentlen,1000)
		ys = kde(xs)
		ys = ys * (first_ys_max/ys.max())
		yss.append(ys)

	pairs = []
	for i in range(len(yss)):
		for j in range(len(yss)):
			if i<j:
				pairs.append((i,j))

	klds = []
	for pair in pairs:
		kld = kl_div(yss[pair[0]],yss[pair[1]])
		klds.append(kld.sum())
	kldmean = sum(klds)/len(klds)
	kldstd = numpy.std(klds)
	return kldmean,kldstd

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
	filelist = os.listdir(corpora_dir)
	for fname in filelist:
		if fname.find('pkl') != -1:
			yearlist.add(int(re.search('\d{4}',fname)[0]))
	yearlist = sorted(yearlist)
	print(yearlist)
	corpora = corpora_loader(yearlist,f'{corpora_dir.joinpath(filelist[0])}')
	res = diff_vwords_vlens_vyears(wordlist,start_len,end_len,corpora,bandwidth)
