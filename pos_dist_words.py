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


def 绘图函数():
	pass



# 函数ID: 2022012030944
# 依赖函数: 2022012030209
# 给posdist画个直方图
def draw_hist_posdist_word(word,corpus_pkl,sent_length,res_dir,lang,**kwargs):
	# 获得数据
	log = posdist_word(word,corpus_pkl,sent_length)['log']
	print(f'totally {len(log)} position records of sents of len {sent_length}')
	# 作图
	# fig = plt.figure(num=1,figsize=(16,9))
	fig = plt.figure(num=1)
	ax1 = fig.subplots()
	binsn = numpy.array([b for b in range(1,sent_length+2)])
	ax1.hist(log,bins=binsn,edgecolor='k',density=True)
	ax1.set_title(f'{word} sent_len: {sent_length}')
	xticks = binsn[:-1]
	plt.xticks(xticks+0.5,xticks,rotation='vertical')
	if kwargs:
		ax1.set_aspect(aspect =kwargs['aspect'])
		plt.title(f'year:{kwargs["year"]} word:{word}({lang}) sent_len:{str(sent_length)}')
		plt.xlabel(kwargs['xlabel'])
		plt.ylabel(kwargs['ylabel'])
	# 保存
	now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	filename = f'{word}_{str(sent_length)}_{lang}_{now}.svg'
	pathfilename = Path(res_dir).joinpath(filename)
	plt.savefig(f'{pathfilename}',format='svg')
	plt.clf()
	return 

# 函数ID: 2022012055139
# 依赖函数: 2022012062617
# 给posdist_multiword画个直方图
def draw_hist_posdist_multiword(corpus_pkl,sent_length,res_dir,lang,*words):
	log = posdist_multiword(corpus_pkl,sent_length,*words)['log']
	plt.clf()
	fig = plt.figure(num=1,figsize=(16,9))
	ax1 = fig.subplots()
	binsn = numpy.array([b for b in range(1,sent_length+2)])
	ax1.hist(log,bins=binsn,edgecolor='k',density=True)
	ax1.set_title(f'{"_".join([i for i in words])} sent_len: {sent_length}')
	xticks = binsn[:-1]
	plt.xticks(xticks+0.5,xticks,rotation='vertical')
	filename = f'{" ".join([i for i in words])}_{str(sent_length)}_{lang}.svg'
	pathfilename = pathfilename = Path(res_dir).joinpath(filename)
	plt.savefig(f'{pathfilename}',format='svg')
	plt.clf()
	return

# 函数ID: 2022012062728
# 依赖函数: 2022012030558
# 函数功能: 给posdist_phrase
def draw_hist_posdist_phrase(corpus_pkl,sent_length,phrase,res_dir,lang,words):
	'''words参数是一个list, 其中的每个元素是一个字符串; phrase参数是一个字符串中间空格分开, 最多支持两个单词的phrase'''
	log = posdist_phrase(corpus_pkl,sent_length,phrase)['log']
	plt.clf()
	fig = plt.figure(num=1,figsize=(16,9))
	ax1 = fig.subplots()
	binsn = numpy.array([b for b in range(1,sent_length+2)])
	ax1.hist(log,bins=binsn,edgecolor='k',density=True)
	ax1.set_title(f'{"_".join([i for i in words])} sent_len: {sent_length}')
	xticks = binsn[:-1]
	plt.xticks(xticks+0.5,xticks,rotation='vertical')
	filename = f'{" ".join([i for i in words])}_{str(sent_length)}_{lang}.svg'
	pathfilename = pathfilename = Path(res_dir).joinpath(filename)
	plt.savefig(f'{pathfilename}',format='svg')
	plt.clf()
	return

# 函数ID: 2022012062802
# 依赖函数: 2022012030944
# 函数功能: 对posdist进行批量可视化
def draw_hist_posdist_word_vlens(word,corpus_pkl,start_len,end_len,res_dir,lang):
	for i in range(start_len,end_len+1):
		res = draw_hist_posdist_word(word,corpus_pkl,i,res_dir,lang)
	return

# 函数ID: 2022012062952
# 依赖函数: 2022012030209
# 函数功能: 给posdist_word话直方图, 多个句长对应多个面板
def draw_hist_posdist_vlens_panels(word,corpus_pkl,start_len,end_len,interval,rows,cols,textclip,lang):
	'''一张图集合多个图像\ninterval是绘图句长间隔, rows是图像行数, cols是图像列数'''
	fig = plt.figure()
	fig.tight_layout()
	axes = fig.subplots(rows,cols).flatten()
	axes_idx = 0
	start_len_right = start_len+interval-1
	for i in range(start_len_right,end_len+1,interval):
		log = posdist_word(word,corpus_pkl,i)['log']
		binsn = numpy.array([b for b in range(1,i+2)])
		axes[axes_idx].hist(log,bins=binsn,edgecolor='k',density=True)
		xticks = binsn[:-1]
		axes[axes_idx].set_xticks(xticks+0.5)
		axes[axes_idx].set_xticklabels(xticks,rotation='vertical')
		axes[axes_idx].set_ylabel('probability')
		axes[axes_idx].set_xlabel('serial-position of target word')
		if textclip == 'left':
			coord1 = 0.025
			coord2 = 0.80
		elif textclip == 'right':
			coord1 = 0.725
			coord2 = 0.80
		axes[axes_idx].text(coord1,coord2,f'Language: {lang}\nTarget Word: {word}\nSentence Length: {i}\nSentence Count: {len(log)}',bbox=dict(boxstyle='Round',facecolor='red', alpha=0.3),transform=axes[axes_idx].transAxes)
		axes_idx += 1
	plt.show()
	return

# 函数ID: 2022012063105
# 依赖函数: 2022012062728
# 功能: 函数2022012062728的多句长版
def draw_hist_posdist_phrase_vlens(corpus_pkl,start_len,end_len,phrase,res_dir,lang,words):
	'''words参数是一个list, 其中的每个元素是一个字符串; phrase参数是一个字符串中间空格分开, 最多支持两个单词的phrase'''
	for i in range(start_len,end_len+1):
		res = draw_hist_posdist_phrase(corpus_pkl,i,phrase,res_dir,lang,words)
	return

# 函数ID: 2022012055119
# 依赖函数: 2022012055139
# 函数功能: 绘制posdist_multiword在多句长的多个直方图
def draw_hist_posdist_multiword_vlens(corpus_pkl,start_len,end_len,res_dir,lang,*words):
	for i in range(start_len,end_len+1):
		res = draw_hist_posdist_multiword(corpus_pkl,i,res_dir,lang,*words)

# 函数ID: 2022012055044
# 依赖函数: 2022012030209
# 函数功能: 画posdist_word的折线图
def draw_line_posdist_word(word,corpus_pkl,sent_length):
	log = posdist_word(word,corpus_pkl,sent_length)['log']
	fig = plt.figure(num=1,figsize=(16,9))
	ax1 = fig.subplots()
	binsn = numpy.array([b for b in range(1,sent_length+2)])
	histres = ax1.hist(log,bins=binsn,edgecolor='k',density=True)
	plt.clf()
	fig = plt.figure(num=1,figsize=(16,9))
	ax1 = fig.subplots()
	histys = histres[0].tolist()
	histxs = [i+0.5 for i in range(1,len(histys)+1)]
	ax1.set_title(f'{word} sent_len: {sent_length}')
	xticks = binsn[:-1]
	ax1.plot(histxs,histys,ls='--',marker='^',fillstyle='none')
	plt.xticks(xticks+0.5,xticks,rotation='vertical')
	plt.show()
	return

# 函数ID: 2022012043223
# 依赖函数: 2022012030209
# 函数功能: 绘制同一单词不同长度的位频分布折线图
# 示例用法:
	# draw_line_posdist_word_vlens('at',corpus_en,12,36,4,xlabel='linear position of word in sentence',ylabel='probability',title='at (English)')
def draw_line_posdist_word_vlens(word,corpus,start_len,end_len,interval,**kwargs):
	style_cnt = 0
	markerstyle = ['+','x','*','D','s','o','^','v']
	markerstyle_sample_size = int((end_len-start_len)/interval)+1
	if markerstyle_sample_size > len(markerstyle):
		print(f'Not enough marker style, support no more than {len(markerstyle)}, but get {markerstyle_sample_size}')
	markerstyle_sample = random.sample(markerstyle,markerstyle_sample_size)
	for length in range(start_len,end_len+1,interval):
		ys = posdist_word(word,corpus,length)['fd_density']
		xs = [i+1 for i in range(length)]
		plt.plot(xs,ys,label=str(length),marker=markerstyle_sample[style_cnt],markersize=4,fillstyle='none',linewidth=0.8,linestyle=(0,(6,style_cnt)))
		style_cnt += 1
	try:
		plt.xlabel(kwargs['xlabel'])
		plt.ylabel(kwargs['ylabel'])
		plt.title(kwargs['title'])
	except:
		pass
	xticks = [i for i in range(1,end_len+1)]
	plt.xticks(xticks,xticks,rotation='vertical')
	plt.legend(title = 'Sent Len')
	plt.show()
	return

# 函数ID: 2022012063431
# 依赖函数: 2022012030209
# 函数功能: 绘制同一单词不同长度的位频分布折线图(使用预计算数据)
# 示例用法:
	# draw_line_posdist_word_precalc_vlens('at',corpus_en,12,36,4,xlabel='linear position of word in sentence',ylabel='probability')
# 参数解释
	# posdist_density_database: 函数2022012030218的产出
def draw_line_posdist_word_vlens_precalc(word,posdist_density_database,start_len,end_len,interval,**kwargs):
	style_cnt = 0
	markerstyle = ['+','x','*','D','s','o','^','v']
	markerstyle_sample_size = int((end_len-start_len)/interval)+1
	if markerstyle_sample_size > len(markerstyle):
		print(f'Not enough marker style, support no more than {len(markerstyle)}, but get {markerstyle_sample_size}')
	markerstyle_sample = random.sample(markerstyle,markerstyle_sample_size)
	for length in range(start_len,end_len+1,interval):
		ys = posdist_density_database[length][word]
		xs = [i+1 for i in range(length)]
		plt.plot(xs,ys,label=str(length),marker=markerstyle_sample[style_cnt],markersize=4,fillstyle='none',linewidth=0.8,linestyle=(0,(6,style_cnt)))
		style_cnt += 1
	try:
		plt.xlabel(kwargs['xlabel'])
		plt.ylabel(kwargs['ylabel'])
		plt.title(kwargs['title'])
	except:
		pass
	xticks = [i for i in range(1,end_len+1)]
	plt.xticks(xticks,xticks,rotation='vertical')
	plt.legend(title = 'Sent Len')
	plt.show()
	return





# 函数ID: 2022012055029
# 依赖函数: 2022012030209
# 函数功能: 绘制同一单词不同长度的KDE平滑曲线图, 各曲线对齐到第一个长度曲线的y轴上限
def draw_curv_posdist_word_vlens(word,corpus_pkl,start_len,end_len,bandwidth,interval):
	'''bandwidth控制KDE曲线的平滑程度, 越大越平滑, 但是越失真, 一般取1\ninterval决定隔多少句长间隔画一次线'''
	first_log = posdist_word(word,corpus_pkl,start_len)['log']
	first_kde = stats.gaussian_kde(first_log,bw_method=bandwidth)
	first_xs = numpy.linspace(1,start_len,1000)
	first_ys = first_kde(first_xs)
	first_ys_max = first_ys.max()
	fig = plt.figure()
	ax = fig.subplots()
	yss = []
	start_len = start_len+interval-1
	for i in range(start_len,end_len+1,interval):
		# 获得数据
		log = posdist_word(word,corpus_pkl,i)['log']
		# 作图
		kde = stats.gaussian_kde(log,bw_method=bandwidth)
		xs = numpy.linspace(1,i,1000)
		ys = kde(xs)
		ys = ys * (first_ys_max/ys.max())
		yss.append(ys)
		xshift = xs-1
		xscale = (xs-1) * ((end_len-1)/(i-1))
		xscale = xscale+1
		ax.plot(xscale,ys,label=str(i),linewidth=0.75)
		plt.legend()
	ax.set_xticks([i for i in range(end_len+1)])
	plt.show()
	return





# 函数ID: 20220121174630
# 参数解释:
# 	stats_path: 144个head的准确度数据, all_heads_accs_20220121174617.pkl
# 	baseline: position probe的local majority baseline, 数据类型float, 论文中是0.153942
def draw_scat_line_probe_acc(stats_path,save_path,baseline):
	fig = plt.figure(dpi=300)
	ax = fig.subplots()
	with open(stats_path,mode=r'rb') as file:
		stats = pickle.load(file)
	for i in range(12):
		ax.scatter([i+1]*12,stats[i],s=10)
	lay_acc_means = [sum(i)/12 for i in stats]
	print(lay_acc_means)
	ax.plot(range(1,13),lay_acc_means,marker='^')
	ax.plot([1,12],[baseline,baseline],linewidth=0.8,ls='--',color='k')
	ax.text(3.5,0.1,f'local majority baseline: {baseline}')
	ax.set_xlabel('attention layer')
	ax.set_ylabel('accuracy')
	ax.set_xticks(range(1,13))
	ax.set_ylim([0,1.05])
	# plt.grid()
	# plt.show()
	now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	plt.savefig(save_path+'\\'+now)
	return
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


# 函数ID: 2022012225031
# 注意: 本函数的前置代码必须使用, 是输入的参数
# 依赖函数: posdist_word
# 函数功能: 绘制同一单词同一长度不同年份的位频分布折线图 (3种语言) [文中使用]
# 前置代码↓ (内存使用63.3 GB) (已检查可用20220402145950)
# yearlist_en = [2005,2009,2013,2017]
# yearlist_de = [2007,2011,2015,2019]
# yearlist_es = [2008,2012,2016,2020]
# yearlists = [yearlist_en,yearlist_de,yearlist_es]
# yearlist_en_ref = [2005,2006,2007,2008,2009,2010,2013,2014,2015,2016,2017,2018,2019,2020]
# yearlist_de_ref = [2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]
# yearlist_es_ref = [2006,2007,2008,2009,2010,2011,2012,2013,2016,2017,2018,2019,2020]
# yearlists_ref = [yearlist_en_ref,yearlist_de_ref,yearlist_es_ref]
# corpora_en = posdist_new.corpora_loader(yearlist_en_ref,r'D:\thehow\3\POSDIST\CORPUS\2_CORPUS\EN\CODENRES\RES\YEAR\eng_news_2005_1M-sentences.pkl') # 这里没有错, 具体年份会被corpora_loader函数替换掉
# corpora_de = posdist_new.corpora_loader(yearlist_de_ref,r'D:\thehow\3\POSDIST\CORPUS\2_CORPUS\DE\CODENRES\RES\YEAR\deu_news_2005_1M-sentences.pkl') # 这里没有错, 具体年份会被corpora_loader函数替换掉
# corpora_es = posdist_new.corpora_loader(yearlist_es_ref,r'D:\thehow\3\POSDIST\CORPUS\2_CORPUS\ES\CODENRES\RES\YEAR\spa_news_2006_1M-sentences.pkl') # 这里没有错, 具体年份会被corpora_loader函数替换掉
# corpora = [corpora_en,corpora_de,corpora_es]
# words = ['but', 'auf', 'los']
# titles = ['but (English)', 'auf (German)', 'los (Spanish)']
# 示例用法: posdist_new.draw_line_posdist_word_len_years_panels(3,1,20,words,titles,corpora,yearlists,yearlists_ref,'png')
def draw_line_posdist_word_len_years_panels(rows,cols,sent_length,words,titles,corpora,year_lists,yearlists_ref,img_fmt):
	fig = plt.figure(figsize=(5,15),dpi=300)
	axes = fig.subplots(rows,cols).flatten()
	word_cnt = 0
	for word in words:
		for year in year_lists[word_cnt]:
			corpus_idx = yearlists_ref[word_cnt].index(year)
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
def draw_line_posdist_meaning_langs_precalc_panels(wordlists,legends,sent_length,corpora_databases):
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
	plt.savefig(f'D:/{now}.png',format='png')
	return

# 函数ID: 20220121225359
# 依赖函数: 无
# 函数功能: 绘制6种语言的句长分布
# 函数参数:
#	langs_list: 样例 ['EN','DE','FR','ES','RU','CZ']
# 	legends: 样例 ['English', 'German', 'French', 'Spanish', 'Russian', 'Czech']
# 	langs_list须和legends_list一一对应
#	pkl_path: 样例 D:\thehow\3\POSDIST\CORPUS\9_SENT_LENS\EN_ALL_YEAR.pkl
# 用例: posdist_new.draw_line_langs_sentlen_dists(lang_list,legends,r'D:\thehow\3\POSDIST\CORPUS\9_SENT_LENS\EN_ALL_YEAR.pkl',r'D:/')
# 注意事项: 函数对文件路径高度敏感, 可迁移性极差
def draw_line_langs_sentlen_dists(lang_list,legends,pkl_path,save_path):
	fig = plt.figure(dpi=300)
	ax = fig.subplots()
	for i in range(len(lang_list)):
		lang_pkl_path = pkl_path[:-15]+lang_list[i]+pkl_path[-13:]
		with open(lang_pkl_path,mode='rb') as file:
			sentlen_pickle = pickle.load(file)
			sentlen_length = len(sentlen_pickle)
			sentlen_freqdist = Counter(sentlen_pickle)
			sentlen_freqdist_sorted = sorted(sentlen_freqdist.items(), key = lambda kv: kv[0])
			sentlen_freqdist_sorted_density = [(k,v/sentlen_length) for k,v in sentlen_freqdist_sorted]
			sentlen_freqdist_sorted_density_50 = sentlen_freqdist_sorted_density[:50]
			xs = [k for k,v in sentlen_freqdist_sorted_density_50]
			ys = [v for k,v in sentlen_freqdist_sorted_density_50]
			ax.plot(xs,ys,label=legends[i],linewidth=0.5,marker='^',markersize=3)
	xticks = [i for i in range(1,51)]
	plt.xticks(xticks,rotation='vertical')
	plt.xlabel('sentence length')
	plt.ylabel('probability')
	plt.legend(title='language')
	# plt.show()
	now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	plt.savefig(save_path+'\\'+now)
	return

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
def diff_word_len_vyears(word,sent_length,corpus_list,bandwidth):
	yss = []
	for corpus in corpus_list:
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
def diff_vwords_vlens_vyears(wordlist,start_len,end_len,corpus_list,bandwidth):
	kldmeans = []
	kldstds = []
	for word in wordlist:
		for length in range(start_len,end_len+1):
			kldmeanstd = diff_word_len_vyears(word,length,corpus_list,bandwidth)
			kldmeans.append(kldmeanstd[0])
			kldstds.append(kldmeanstd[1])
	kldmeans_avg = numpy.mean(kldmeans)
	kldstds_avg = numpy.mean(kldstds)
	return kldmeans_avg,kldstds_avg





























def 弃用函数():
	pass

# 函数ID: 2022012063558
# 依赖函数: 2022012030209
# 函数功能: 计算放缩后的直方图的偏度和峰度, 并可视化为二维散点图
def draw_scat_curve_skewkurt_vwords_vlens(words,corpus_pkl,start_len,end_len,bandwidth,interval):
	'''words是字符串的列表\nbandwidth控制KDE曲线的平滑程度, 越大越平滑, 但是越失真, 一般取1\ninterval决定隔多少句长间隔画一次线'''
	fig = plt.figure()
	ax = fig.subplots()
	for word in words:
		skews = []
		kurts = []
		for i in range(start_len,end_len+1,interval):
			# 获得数据
			log = posdist_word(word,corpus_pkl,i)['log']
			# 作图
			kde = stats.gaussian_kde(log,bw_method=bandwidth)
			xs = numpy.linspace(0,i,1000)
			xscale = xs * (end_len/i)
			ys = kde(xs)
			ys = ys * (i/end_len)
			print(len(ys))
			skews.append(stats.skew(ys))
			kurts.append(stats.kurtosis(ys))
		ax.set_xlim(-1.5,1.5)
		ax.set_ylim(-1.5,1.5)
		ax.scatter(skews,kurts,facecolor=None,s=2)
	plt.show()
	return

# 基于预计算分布的batch_diff_len_vwords
def batch_diff_len_vwords_precount(wordlist,posdist_count,start_len,end_len,bandwidth):
	def len_words(wordlist,posdist_count,start_len,sent_length,bandwidth):
		yss = []
		for word in wordlist:
			first_log = posdist_count[start_len][word]
			first_kde = stats.gaussian_kde(first_log,bw_method=bandwidth)
			first_xs = numpy.linspace(1,start_len,1000)
			first_ys = first_kde(first_xs)
			first_ys_max = first_ys.max()

			log = posdist_count[seng_length][word]
			kde = stats.gaussian_kde(log,bw_method=bandwidth)
			xs = numpy.linspace(1,sent_length,1000)
			ys = kde(xs)
			ys = ys * (first_ys_max/ys.max())
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

	kl_means = []
	kl_stds = []
	for i in range(start_len,end_len+1):
		meanstd = len_words(wordlist,posdist_count,start_len,i,bandwidth)
		kl_means.append(meanstd[0])
		kl_stds.append(meanstd[1])
	kl_mean_means = numpy.mean(kl_means)
	kl_mean_stds = numpy.mean(kl_stds)
	return kl_mean_means,kl_mean_stds



# diff_word_vlens的基于预计算结果的版本
# 逻辑错误, 无法使用, log和fd_cnt在KDE上是不一样的
def diff_word_vlens_precount(word,posdist_count,start_len,end_len,bandwidth):
	# 获得第一个句长的高度, 用于后边句长的对齐
	first_log = posdist_count[start_len][word]
	first_kde = stats.gaussian_kde(first_log,bw_method=bandwidth)
	first_xs = numpy.linspace(1,start_len,1000)
	first_ys = first_kde(first_xs)
	first_ys_max = first_ys.max()

	yss = []
	# 对于每个句长, 计算
	for i in range(start_len,end_len+1,1):
		# 获得数据
		log = posdist_count[i][word]
		# 作图
		kde = stats.gaussian_kde(log,bw_method=bandwidth)
		xs = numpy.linspace(1,i,1000)
		ys = kde(xs)
		ys = ys * (first_ys_max/ys.max()) #对齐到第一个句长的高度
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
	









# (相同句长, 不同单词)场景的平均KL散度均值, 和平均KL散度标准差, 单一句长
def kl_len_words(wordlist,corpus_pkl,sent_length,start_len,end_len,bandwidth):
	yss = []
	for word in wordlist:
		first_log = posdist_word(word,corpus_pkl,start_len)['log']
		first_kde = stats.gaussian_kde(first_log,bw_method=bandwidth)
		first_xs = numpy.linspace(1,start_len,1000)
		first_ys = first_kde(first_xs)
		first_ys_max = first_ys.max()
		
		log = posdist_word(word,corpus_pkl,sent_length)['log']
		kde = stats.gaussian_kde(log,bw_method=bandwidth)
		xs = numpy.linspace(1,sent_length,1000)
		ys = kde(xs)
		ys = ys * (first_ys_max/ys.max())
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






