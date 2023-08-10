import os
import sys
import torch
import warnings
from transformers import BertModel
from transformers import BertTokenizer
from transformers import logging
from pathlib import Path
import pickle
import inspect

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def sent_selector(word,sentlen,corpus):
	wordin_tksents = [tksent for tksent in corpus if word in tksent]
	shorter_tksents = [tksent for tksent in wordin_tksents if sentlen-10<=len(tksent)<=sentlen]
	truelen_sents = []
	for tksent in shorter_tksents:
		shorter_sent = ' '.join(tksent)
		tokenized_sent_obj = bert_tokenizer(shorter_sent)
		tokenized_sent_ids = tokenized_sent_obj['input_ids']
		tokenized_sent_tokens = bert_tokenizer.convert_ids_to_tokens(tokenized_sent_ids)
		double_sharp_count = 0
		for token in tokenized_sent_tokens:
			if token.startswith('##'):
				double_sharp_count += 1
		tokenized_sent_len = len(tokenized_sent_tokens)-2-double_sharp_count
		if tokenized_sent_len == sentlen:
			truelen_sents.append(shorter_sent)
	print(f'[{inspect.stack()[0][3]}] {len(truelen_sents)} sents extracted.')
	return truelen_sents


'''
[函数注释]
	[功能]
		1. 主要功能: 选择含有目标词指定句长的句子
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
		2. sent_length
			1. 数据类型: int
			2. 数据结构: int
			3. 参数类型: 必选
			4. 语义: 指定句长
			5. 取值范围: [1,50]
			6. 获得来源: 手动输入
			7. 样例文件/输入: 36
		3. corpus
			1. 数据类型: list
			2. 数据结构: [['string1', 'string2'],['string3', 'string4']]
			3. 参数类型: 必选
			4. 语义: 语料库
			5. 取值范围:	
			6. 获得来源: 外部加载的pickle
			7. 样例文件/输入: 20230809163014.pkl
	[用例]
		1. space_len_sents_selector('and',36,corpus)
			1. 语句: 
			2. 输出
				1. 语义: 筛选而来的句子
				2. 数据类型: list
				3. 数据结构: [string, string]
				4. 样例文件/输出: 20230809163235.pkl
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
		1.
		2. 
'''
def space_len_sents_selector(word,sent_length,corpus):
	sents_len = [sent for sent in corpus if len(sent) == sent_length]
	sents_len_word = [' '.join(sent) for sent in sents_len if word in sent]
	return sents_len_word

def complete_tokenizer(sent):
	def get_id_map(trimmed_inputids):

		cnt = -1
		idmap_list = [] 
		for tokenid in trimmed_inputids:
			if bert_tokenizer.convert_ids_to_tokens([tokenid])[0].startswith('##'):
				cnt = cnt
			else:
				cnt += 1
			idmap_list.append(cnt)
		idmap_dict = dict()
		for wordid in idmap_list:
			idmap_dict[wordid] = [tokenid for tokenid,realwordid in enumerate(idmap_list) if realwordid==wordid]
		return {'idmap_list': idmap_list, 'idmap_dict': idmap_dict}

	def get_onepiece_tokens(tokenlist,maplist):
		mapdict = dict()
		for i in maplist:
			mapdict[i] = [k for k,v in enumerate(maplist) if v==i]
		tokendict = dict()
		for k in mapdict:
			tokendict[k] = [tokenlist[token] for token in mapdict[k]]
		tokendictj = dict()
		for item in tokendict.items():
			raw_tokens = item[1]
			new_tokens = []
			for j in item[1]:
				if j.startswith('##'):
					new_tokens.append(j[2:])
				else:
					new_tokens.append(j)
			tokendictj[item[0]] = ''.join(new_tokens)
		return tokendictj
	
	tokenized_sent = bert_tokenizer(sent,return_tensors='pt')
	raw_input_ids = tokenized_sent['input_ids'][0]
	trimmed_input_ids = raw_input_ids[1:-1]
	idmap = get_id_map(trimmed_input_ids)
	idmap_list = idmap['idmap_list']
	idmap_dict = idmap['idmap_dict']
	raw_input_tokens = bert_tokenizer.convert_ids_to_tokens(raw_input_ids)
	trimmed_input_tokens = raw_input_tokens[1:-1]
	dense_input_tokens_dict = get_onepiece_tokens(trimmed_input_tokens,idmap_list)
	dense_input_tokens_list = list(dense_input_tokens_dict.values())
	return {'tokenized_sent':tokenized_sent,'raw_input_ids':raw_input_ids,'trimmed_input_ids':trimmed_input_ids,'raw_tokens':raw_input_tokens,'trimmed_tokens':trimmed_input_tokens,'dense_tokens': dense_input_tokens_list,'idmap_list':idmap_list,'idmap_dict':idmap_dict}

def attn_matrix_trim_scale(attn_matrix):
	core_matrix = attn_matrix[1:-1,1:-1]
	container_tensor = torch.zeros(core_matrix.shape)
	row_cnt = 0
	for row in core_matrix:
		mtplr = 1/sum(row)
		scaled_row = row*mtplr
		container_tensor[row_cnt] = scaled_row
		row_cnt += 1
	return container_tensor

def attn_matrix_denser(idmap_dict,trim_scale_attn_matrix):
	def rebuild_row(idmap_dict,trim_scale_attn_matrix):
		new_tensor = torch.zeros((len(idmap_dict.keys()),len(trim_scale_attn_matrix)))
		for row_no_new in idmap_dict.keys():
			new_tensor[row_no_new] = trim_scale_attn_matrix[idmap_dict[row_no_new]].sum(axis=0)/len(idmap_dict[row_no_new])
		return new_tensor
	def rebuild_col(idmap_dict,trim_scale_attn_matrix):
		new_tensor = torch.zeros((len(idmap_dict.keys()),len(idmap_dict.keys())))
		for col_no_new in idmap_dict:
			new_tensor[:,col_no_new] = trim_scale_attn_matrix[:,idmap_dict[col_no_new]].sum(axis=1)
		return new_tensor
	row_proced = rebuild_row(idmap_dict,trim_scale_attn_matrix)
	col_proced = rebuild_col(idmap_dict,row_proced)
	matrix_proced = col_proced
	return matrix_proced

def get_word_position(word,sent):
	tokenized_sent = bert_tokenizer(sent,return_tensors='pt')
	input_ids = tokenized_sent['input_ids'][0]
	input_ids = input_ids[1:-1]

	better_token = complete_tokenizer(sent)
	token_word_id_map_list = better_token['idmap_list']
	token_word_id_map_dict = better_token['idmap_dict']
	
	target_word_id = bert_tokenizer.convert_tokens_to_ids(word)
	target_word_attn_row_num_token =  (input_ids == target_word_id).nonzero()[0]
	target_word_attn_row_num_dense = token_word_id_map_list[target_word_attn_row_num_token]

	return target_word_attn_row_num_dense

def concat_len_sents_selector(word,sent_len,corpus,sent_max):
	sents = space_len_sents_selector(word,sent_len,corpus)
	sents += space_len_sents_selector(word,sent_len-1,corpus)
	sents += space_len_sents_selector(word,sent_len-2,corpus)
	sents += space_len_sents_selector(word,sent_len-3,corpus)
	truelen_sents = []
	limit_cnt = 1
	for sent in sents:
		if limit_cnt <=sent_max:
			res_better_token  = complete_tokenizer(sent)
			true_len  = len(res_better_token['dense_tokens'])
			if true_len == sent_len:
				truelen_sents.append(sent)
				limit_cnt += 1
		else:
			break
	return truelen_sents

def get_word_attn_rowlabs(word,sent_len,corpus,sent_max):
	word_pos = []
	truelen_sents = concat_len_sents_selector(word,sent_len,corpus,sent_max)
	len_truelen_sents = len(truelen_sents)
	attn_rowlabs = torch.zeros(12,12,len_truelen_sents,sent_len+1)

	tsent_cnt = 0
	for tsent in truelen_sents:
		tokenized_sent = bert_tokenizer(tsent,return_tensors='pt')
		res_better_token  = complete_tokenizer(tsent)
		idmap_dict = res_better_token['idmap_dict']
		idmap_list = res_better_token['idmap_list']
		sent_attn_144 = torch.stack(bert_model(**tokenized_sent,output_attentions=True)['attentions']).squeeze()
		word_row_num = get_word_position(word,tsent)
		word_pos.append(word_row_num)
		for layer in range(12):
			for head in range(12):
				attn_trimmed = attn_matrix_trim_scale(sent_attn_144[layer][head])
				attn_dense = attn_matrix_denser(idmap_dict,attn_trimmed)
				attn_row = attn_dense[word_row_num].detach()
				attn_rowlabs[layer][head][tsent_cnt][:-1]=attn_row
				attn_rowlabs[layer][head][tsent_cnt][-1] = word_row_num
		tsent_cnt += 1
	return attn_rowlabs

def get_words_attn_rowlabs(wordlist,sent_len,corpus,sent_max,save_path):
	words_attnrowlabs144 = []
	words_line_cnt = []
	for word in wordlist:
		print(f'Extracting attention rows for word \"{word}\"...')
		oneword_attnrowlabs = get_word_attn_rowlabs(word,sent_len,corpus,sent_max)
		words_attnrowlabs144.append(oneword_attnrowlabs)
		word_line_cnt = oneword_attnrowlabs.shape[2]
		words_line_cnt.append(word_line_cnt)
		print(f'Attention weights and position extracted for word {word} in {word_line_cnt} length-{sent_len} sentences')

	words_line_cnt_sum = sum(words_line_cnt)
	heads_attnrowlabs = torch.zeros(12,12,words_line_cnt_sum,sent_len+1)
	for layer in range(12):
		for head in range(12):
			onehead_attnrowlabs = []
			for attnrowlabs144 in words_attnrowlabs144:
				onehead_attnrowlabs.append(attnrowlabs144[layer][head])
			onehead_attnrowlabs = torch.cat(onehead_attnrowlabs)
			heads_attnrowlabs[layer][head] = onehead_attnrowlabs
			filename = Path(save_path).joinpath(f'{layer+1:02d}_{head+1:02d}').joinpath('data').joinpath(f'attnrowlabs_{layer+1:02d}_{head+1:02d}.pkl')
			os.makedirs(Path(save_path).joinpath(f'{layer+1:02d}_{head+1:02d}').joinpath('data'),exist_ok=False)
			with open(filename,mode='wb') as pkl:
				pickle.dump(onehead_attnrowlabs,pkl)
	print(f'Attention rows and positions saved at {save_path} ')
	return heads_attnrowlabs

if __name__ == '__main__':
	print('Loading BERT...')
	sent_len = int(sys.argv[1])
	sent_max = int(sys.argv[2])
	corpus_path = Path(sys.argv[3])
	save_path = Path(sys.argv[4])
	wordlist = sys.argv[5].split()
	print('Loading corpus...')
	with open(corpus_path, mode='rb') as pkl:
		corpus = pickle.load(pkl)
	print('Extracting attention rows...')
	res = get_words_attn_rowlabs(wordlist,sent_len,corpus,sent_max,save_path)