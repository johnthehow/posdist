# coding=utf8
import re
import os
import sys
import string
import pickle
from pathlib import Path
import warnings
import bs4
warnings.filterwarnings("ignore")



def replace_gibberish_doc(doc):
	hex_map = GIBBERISH_TABLE
	hex_map_keys = list(hex_map.keys()) 
	for key in hex_map_keys:
		doc = doc.replace(key,hex_map[key])
	return doc

def replace_gibberish_strlines(strlines):
	proc_lines = []
	hex_map = GIBBERISH_TABLE
	hex_map_keys = list(hex_map.keys()) 
	for line in strlines:
		for key in hex_map_keys:
			line = line.replace(key,hex_map[key])
		proc_lines.append(line)
	return proc_lines


def remove_extra_space(strlines):
	proc_lines = []
	for line in strlines:
		proc_line = re.sub(' {2,}',' ',line)
		proc_lines.append(proc_line)
	return proc_lines


def log_illegal_token(tklines):
	rec = []
	for line in tklines:
		for token in line:
			illegal_cnt = 0
			for char in token:
				if char not in LEGAL_CHAR:
					illegal_cnt += 1
			if illegal_cnt != 0:
				rec.append(token)
	res = '\n'.join(rec)
	return res


def replace_html_char_entity(strlines):
	proc_lines = []
	for line in strlines:
		proc_lines.append(bs4.BeautifulSoup(line).text)
	return proc_lines


def remove_empty_strlines(strlines):
	res = []
	empty_line_cnt = 0
	for line in strlines:
		if len(line) != 0:
			res.append(line)
		else:
			empty_line_cnt += 1
	return res


def remove_empty_tklines(tklines):
	res = []
	empty_line_cnt = 0
	for line in tklines:
		if len(line) != 0:
			res.append(line)
		else:
			empty_line_cnt += 1
	print(f'{empty_line_cnt} Empty Lines Removed')
	return res

def is_contain_illegal_char(token):
	res = False
	for char in token:
		if char not in LEGAL_CHAR:
			res = True
	return res

def is_only_legalabc(token):
	match_cnt = 0
	for char in token:
		if char in LEGAL_ABC:
			match_cnt += 1
	if match_cnt == len(token) and len(token) != 0:
		res = True
	else:
		res = False
	return res


def is_only_numeric(token):
	match_cnt = 0
	for char in token:
		if char in LEGAL_DIGIT:
			match_cnt += 1
	if match_cnt == len(token) and len(token) != 0:
		res = True
	else:
		res = False
	return res


def is_only_nonvocal(token):
	match_cnt = 0
	for char in token:
		if char not in LEGAL_ABC and char not in LEGAL_DIGIT:
			match_cnt += 1
	if match_cnt == len(token) and len(token) != 0:
		res = True
	else:
		res = False
	return res


def is_homo(token):
	return is_only_legalabc(token) or is_only_numeric(token) or is_only_nonvocal(token)  and len(token) != 0


def is_mix_legalabc_nonvocal(token):
	abc_cnt = 0
	nonvocal_cnt = 0
	for char in token:
		if char in LEGAL_ABC:
			abc_cnt += 1
		elif char not in VOCAL_CHAR:
			nonvocal_cnt += 1
	if abc_cnt + nonvocal_cnt == len(token) and abc_cnt*nonvocal_cnt and len(token) != 0:
		res = True
	else:
		res = False
	return res


def is_mix_legalabc_num(token):
	abc_cnt = 0
	num_cnt = 0
	for char in token:
		if char in LEGAL_ABC:
			abc_cnt += 1
		elif char in LEGAL_DIGIT:
			num_cnt += 1
	if abc_cnt + num_cnt == len(token) and abc_cnt*num_cnt and len(token) !=0 :
		res = True
	else:
		res = False
	return res

def is_mix_num_nonvocal(token):
	num_cnt = 0
	nonvocal_cnt = 0
	for char in token:
		if char in LEGAL_DIGIT:
			num_cnt += 1
		elif char not in LEGAL_ABC:
			nonvocal_cnt += 1
	if num_cnt + nonvocal_cnt == len(token) and num_cnt*nonvocal_cnt and len(token) !=0:
		res = True
	else:
		res = False
	return res


def is_mix_legalabc_num_nonvocal(token):
	res = False
	if is_homo(token) == False:
		if is_mix_legalabc_num(token) == False:
			if is_mix_legalabc_nonvocal(token) == False:
				if is_mix_num_nonvocal(token) == False:
					if len(token) != 0:
						res = True
	return res


def is_contain_onlyone_legalabc_seq(token):
	char_cnt = 0
	abc_cnt = 0
	abc_idxs = []
	for char in token:
		char_cnt += 1
		if char in LEGAL_ABC:
			abc_cnt += 1
			abc_idxs.append(char_cnt)
	res = all(a+1==b for a, b in zip(abc_idxs, abc_idxs[1:])) and abc_cnt !=0
	return res

def keep_only_legalabc_seq(token):
	if is_contain_onlyone_legalabc_seq(token):
		proc_token = ''
		for char in token:
			if char in LEGAL_ABC:
				proc_token += char
	else:
		print('keep_only_legalabc_seq: contain more than one continuous abc sequence')
	return proc_token


def is_contain_sota(token):
	regex = re.compile(f'[{LEGAL_ABC}]+(-[{LEGAL_ABC}]+)+')
	if re.search(regex,token) != None:
		res = True
	else:
		res = False
	return res

def keep_only_sota(token):
	if is_contain_sota(token):
		regex = re.compile(f'[{LEGAL_ABC}]+(-[{LEGAL_ABC}]+)+')
		match = re.search(regex,token)
		res = match.group()
	else:
		print('keep_only_sota: trying to extract sota while don\'t have one')
	return res

def is_usa(token):
	regex = re.compile(f'^[{LEGAL_ABC}](\.[{LEGAL_ABC}])+\.?$')
	if re.search(regex,token) != None:
		res = True
	else:
		res = False
	return res

def is_abc_comma_abc(token):
	regex = re.compile(f'^([{LEGAL_ABC}]{{2,}})(,)([{LEGAL_ABC}]{{2,}})$')
	if re.search(regex,token) != None:
		res = True
	else:
		res = False
	return res


def split_abc_comma_abc(token):
	if is_abc_comma_abc(token):
		res = []
		regex = re.compile(f'^([{LEGAL_ABC}]{{2,}})(,)([{LEGAL_ABC}]{{2,}})$')
		match = re.search(regex,token)
		res.append(match.group(1))
		res.append(match.group(3))
	else:
		print('split_abc_comma_abc: trying to split abc,abc while is not one')
	return res

def is_abc_endpunct_abc(token):
	regex = re.compile(f'^([{LEGAL_ABC}]{{2,}})({END_PUNCT})([{LEGAL_ABC}]{{2,}})$')
	if re.search(regex,token) != None:
		res = True
	else:
		res = False
	return res

def split_abc_endpunct_abc(token):
	if is_abc_endpunct_abc(token):
		res = []
		regex = re.compile(f'^([{LEGAL_ABC}]{{2,}})({END_PUNCT})([{LEGAL_ABC}]{{2,}})$')
		match = re.search(regex,token)
		res.append(match.group(1))
		res.append(match.group(3))
	else:
		print('split_abc_endpunct_abc: trying to split abc.!?abc while is not one')
	return res

def is_contain_contract(token):
	res = False
	regex = re.compile(f'[{LEGAL_ABC}]+\'[{LEGAL_ABC}]+')
	if re.search(regex,token) != None:
		if len(re.findall('\'',token)) == 1:
			res = True
	return res

def keep_only_contract(token):
	regex = re.compile(f'[{LEGAL_ABC}]+\'[{LEGAL_ABC}]+')
	if is_contain_contract(token) == True:
		match = re.search(regex,token).group()
	return match

def strip_nonvocal_mix_nonvocal(token):
	origin = token
	if token[0] not in VOCAL_CHAR:
		token = token[1:]
	else:
		pass
	if token[-1] not in VOCAL_CHAR:
		token = token[:-1]
	else:
		pass
	if token == origin:
		pass
	else:
		token = strip_nonvocal_mix_nonvocal(token)
	return token

def replace_nonascii_punct(strlines):
	nonvocal_conv_table = [('“”„‟⹂❞❝〞〝〟«»❠🙷🙶🙸＂「」⸗	','"'),('‹›’‘‛❛❜❟′ʻʾ´`ʼʿ̕ʽ՝̒̔︐´＇᠈̒','\''),('‒—–―‑‐ー−─¬―—–‐－‑-﹣⁃᠆‧⹀⸚゠֊͜','-'),('…⋯᠁','...'),('⸴̦،🄊⹁⍪⹌⸲‚꛵𖺗𝪇🄁',','),('⁄∕̷⼃','/')]
	proc_lines = []
	for line in strlines:
		proc_line = ''
		for char in line:
			proc_char = char
			for pair in nonvocal_conv_table:
				if char in pair[0]:
					proc_char = pair[1]
				else:
					pass
			proc_line += proc_char
		proc_lines.append(proc_line)
	return proc_lines

def is_contain_url(token):
	url = '\.com|\.net|\.org|\.edu|\.gov|\.mil|\.aero|\.asia|\.biz|\.cat|\.coop|\.info|\.int|\.jobs|\.mobi|\.museum|\.name|\.post|\.pro|\.tel|\.travel|\.xxx|\.ac|\.ad|\.ae|\.af|\.ag|\.ai|\.al|\.am|\.an|\.ao|\.aq|\.ar|\.as|\.at|\.au|\.aw|\.ax|\.az|\.ba|\.bb|\.bd|\.be|\.bf|\.bg|\.bh|\.bi|\.bj|\.bm|\.bn|\.bo|\.br|\.bs|\.bt|\.bv|\.bw|\.by|\.bz|\.ca|\.cc|\.cd|\.cf|\.cg|\.ch|\.ci|\.ck|\.cl|\.cm|\.cn|\.co|\.cr|\.cs|\.cu|\.cv|\.cx|\.cy|\.cz|\.dd|\.de|\.dj|\.dk|\.dm|\.do|\.dz|\.ec|\.ee|\.eg|\.eh|\.er|\.es|\.et|\.eu|\.fi|\.fj|\.fk|\.fm|\.fo|\.fr|\.ga|\.gb|\.gd|\.ge|\.gf|\.gg|\.gh|\.gi|\.gl|\.gm|\.gn|\.gp|\.gq|\.gr|\.gs|\.gt|\.gu|\.gw|\.gy|\.hk|\.hm|\.hn|\.hr|\.ht|\.hu|\.id|\.ie|\.il|\.im|\.in|\.io|\.iq|\.ir|\.is|\.it|\.je|\.jm|\.jo|\.jp|\.ke|\.kg|\.kh|\.ki|\.km|\.kn|\.kp|\.kr|\.kw|\.ky|\.kz|\.la|\.lb|\.lc|\.li|\.lk|\.lr|\.ls|\.lt|\.lu|\.lv|\.ly|\.ma|\.mc|\.md|\.me|\.mg|\.mh|\.mk|\.ml|\.mm|\.mn|\.mo|\.mp|\.mq|\.mr|\.ms|\.mt|\.mu|\.mv|\.mw|\.mx|\.my|\.mz|\.na|\.nc|\.ne|\.nf|\.ng|\.ni|\.nl|\.no|\.np|\.nr|\.nu|\.nz|\.om|\.pa|\.pe|\.pf|\.pg|\.ph|\.pk|\.pl|\.pm|\.pn|\.pr|\.ps|\.pt|\.pw|\.py|\.qa|\.re|\.ro|\.rs|\.ru|\.rw|\.sa|\.sb|\.sc|\.sd|\.se|\.sg|\.sh|\.si|\.sj|\. Ja|\.sk|\.sl|\.sm|\.sn|\.so|\.sr|\.ss|\.st|\.su|\.sv|\.sx|\.sy|\.sz|\.tc|\.td|\.tf|\.tg|\.th|\.tj|\.tk|\.tl|\.tm|\.tn|\.to|\.tp|\.tr|\.tt|\.tv|\.tw|\.tz|\.ua|\.ug|\.uk|\.us|\.uy|\.uz|\.va|\.vc|\.ve|\.vg|\.vi|\.vn|\.vu|\.wf|\.ws|\.ye|\.yt|\.yu|\.za|\.zm|\.zw|\.php|\.html|\.htm|\.asp|http|https|ftp|www|://'
	if re.search(url,token) != None:
		res = True
	else:
		res = False
	return res

def is_contain_currency(token):
	regex = re.compile('[$£€¥₹](\d+)?(,)?(\d+)?(\.?)(\d)+')
	if re.search(regex,token) != None:
		res = True
	else:
		res = False
	return res

def keep_only_currency(token):
	regex = re.compile('[$£€¥₹](\d+)?(,)?(\d+)?(\.?)(\d)+')
	match = re.search(regex,token)
	res = match.group()
	return res

def replace_url(token):
	if is_contain_url(token):
		res = '[URL]'
	return res

def lower_token(tklines):
	proc_lines = []
	sent_cnt = 0
	for line in tklines:
		sent_cnt += 1
		proc_line = []
		for token in line:
			proc_line.append(token.lower())
		proc_lines.append(proc_line)
	return proc_lines

def load_leipzig(filename):
	print(f'Reading Leipzig sentences...')
	with open(filename,mode='r',encoding='utf-8') as lpzfile:
		doc = lpzfile.read()
	print(f'Document length: {len(doc)} chars')

	print(f'Replacing gibberish...')
	doc = replace_gibberish_doc(doc)
	doc = replace_gibberish_doc(doc)
	doc = replace_gibberish_doc(doc)
	print(f'Document length (gibberish replaced): {len(doc)} chars')

	print(f'Splitting lines...')
	strlines = doc.split(sep='\n')
	
	strlines = strlines[:-1]
	print(f'No. of lines (as strings) {len(strlines)} lines')

	print(f'Revmoing line numbers...')
	strlines = [re.sub('^\d+\t','',line) for line in strlines]

	print(f'Replacing HTML entities...')
	strlines = replace_html_char_entity(strlines)
	print(f'No. of lines (HTML replaced): {len(strlines)} lines')

	print(f'Replacing gibberish...')
	strlines = replace_gibberish_strlines(strlines)
	strlines = replace_gibberish_strlines(strlines)
	strlines = replace_gibberish_strlines(strlines)
	print(f'Document length (gibberish replaced): {len(" ".join(strlines))} chars')

	print(f'Replacing non-ascii punctuations...')
	strlines = replace_nonascii_punct(strlines)

	
	print(f'Removing extra spaces...')
	strlines = remove_extra_space(strlines)
	
	print(f'Removing empty lines (as strings)...')
	strlines = remove_empty_strlines(strlines)
	print(f'No. of lines (empty lines removed) {len(strlines)} lines')

	tklines = [line.split() for line in strlines]
	return tklines

def sanitize_token(tklines):
	proc_lines = []
	for line in tklines:
		proc_line = []
		for token in line:
			if len(token) == 0:
				pass
			else:
				if is_contain_illegal_char(token) == True:
					pass
				else:
					if is_homo(token) == True:
						if is_only_legalabc(token) == True:
							proc_line.append(token)
						elif is_only_numeric(token) == True:
							proc_line.append(token)
						elif is_only_nonvocal(token) == True:
							continue
					else:
						if is_mix_legalabc_nonvocal(token) == True:
							if is_contain_onlyone_legalabc_seq(token) == True:
								proc_line.append(keep_only_legalabc_seq(token))
							elif is_contain_sota(token):
								proc_line.append(keep_only_sota(token))
							elif is_usa(token):
								proc_line.append(strip_nonvocal_mix_nonvocal(token))
							elif is_abc_comma_abc(token):
								proc_line += split_abc_comma_abc(token)
							elif is_contain_url(token):
								proc_line.append(replace_url(token))
							elif is_abc_endpunct_abc(token):
								proc_line += split_abc_endpunct_abc(token)
							elif is_contain_contract(token):
								proc_line.append(keep_only_contract(token))
							else:
								proc_line.append(strip_nonvocal_mix_nonvocal(token))
						elif is_mix_legalabc_num(token):
							proc_line.append(token)
						elif is_mix_num_nonvocal(token):
							if is_contain_currency(token):
								proc_line.append(keep_only_currency(token))
							else:
								proc_line.append(strip_nonvocal_mix_nonvocal(token))
						elif is_mix_legalabc_num_nonvocal(token):
							if is_contain_url(token):
								proc_line.append(replace_url(token))
							elif is_contain_onlyone_legalabc_seq(token) == True:
								proc_line.append(keep_only_legalabc_seq(token))
							else:
								proc_line.append(strip_nonvocal_mix_nonvocal(token))
						else:
							print(f'impossible token {token} detected, logical problem in sanitizer.')
		proc_lines.append(proc_line)
	return proc_lines

def stats(tklines):
	stat_log = []
	token_cnt = 0
	num_token_cnt = 0
	abc_token_cnt = 0
	abc_num_token_cnt = 0
	abc_nonvocal_token_cnt = 0
	num_nonvocal_token_cnt = 0
	abc_num_nonvocal_token_cnt = 0
	for line in tklines:
		for token in line:
			token_cnt += 1
			if is_only_legalabc(token) == True:
				abc_token_cnt += 1
			elif is_only_numeric(token) == True:
				num_token_cnt += 1
			elif is_mix_legalabc_num(token) == True:
				abc_num_token_cnt += 1
			elif is_mix_legalabc_nonvocal(token) == True:
				abc_nonvocal_token_cnt += 1
			elif is_mix_num_nonvocal(token) == True:
				num_nonvocal_token_cnt += 1
			else:
				abc_num_nonvocal_token_cnt += 1
	print(f'No. of tokens: \t{token_cnt}')
	stat_log.append(f'No. of tokens: \t{token_cnt}')
	print(f'No. of illegal tokens: \t{len(illegal_tokens)}')
	stat_log.append(f'No. of illegal Tokens: \t{len(illegal_tokens)}')
	print(f'No. of lines (processed): \t{len(tklines)}')
	stat_log.append(f'No. of lines (processed): \t{len(tklines)}')
	print(f'Abc tokens: \t{abc_token_cnt}({100*abc_token_cnt/token_cnt:.3f}%)')
	stat_log.append(f'Abc tokens: \t{abc_token_cnt}({100*abc_token_cnt/token_cnt:.3f}%)')
	print(f'Abc + Num tokens: \t{abc_num_token_cnt}({100*abc_num_token_cnt/token_cnt:.3f}%)')
	stat_log.append(f'Abc + Num tokens: \t{abc_num_token_cnt}({100*abc_num_token_cnt/token_cnt:.3f}%)')
	print(f'Abc + Sym tokens: \t{abc_nonvocal_token_cnt}({100*abc_nonvocal_token_cnt/token_cnt:.3f}%)')
	stat_log.append(f'Abc + Sym tokens: \t{abc_nonvocal_token_cnt}({100*abc_nonvocal_token_cnt/token_cnt:.3f}%)')
	print(f'Abc + Num + Sym tokens: \t{abc_num_nonvocal_token_cnt}({100*abc_num_nonvocal_token_cnt/token_cnt:.3f}%)')
	stat_log.append(f'Abc + Num + Sym tokens: \t{abc_num_nonvocal_token_cnt}({100*abc_num_nonvocal_token_cnt/token_cnt:.3f}%)')
	print(f'Num tokens: \t{num_token_cnt}({100*num_token_cnt/token_cnt:.3f}%)')
	stat_log.append(f'Num tokens: \t{num_token_cnt}({100*num_token_cnt/token_cnt:.3f}%)')
	print(f'Num + Sym tokens: \t{num_nonvocal_token_cnt}({100*num_nonvocal_token_cnt/token_cnt:.3f}%)')
	stat_log.append(f'Num + Sym tokens: \t{num_nonvocal_token_cnt}({100*num_nonvocal_token_cnt/token_cnt:.3f}%)')
	return stat_log

def pipeline(filename,savepath):
	filenamepobj = Path(filename)
	savepathpobj = Path(savepath)
	leipzig_sents_filename = filenamepobj.name
	print('\n')
	print(f'Processing file {filenamepobj.stem}')
	print(f"====================PRE-PROCESS({LANG})====================")
	print(f'Loading Leipzig corpora...')
	proc_lines = load_leipzig(filename)
	print(f'Logging illegal tokens...')
	global illegal_tokens
	illegal_tokens = log_illegal_token(proc_lines)
	print(f'Sanitizing tokens...')
	proc_lines = sanitize_token(proc_lines)
	print(f'Lowering tokens...')
	proc_lines = lower_token(proc_lines)
	print(f'Removeing empty tokenized lines')
	proc_lines = remove_empty_tklines(proc_lines)
	print(f"====================SUMMARY({LANG})====================")
	stat = stats(proc_lines)
	print(f"====================SAVE({LANG})====================")
	try:
		with open(f'{savepathpobj.joinpath(filenamepobj.stem)}.pkl',mode='wb') as save_pkl:
			pickle.dump(proc_lines,save_pkl)
			print(f'Result pickle saved at: {savepathpobj.joinpath(filenamepobj.stem)}.pkl')
	except:
		print('Failed saving result pickle file.')
	try:
		with open(f'{savepathpobj.joinpath(filenamepobj.stem)}_except.txt',mode='w',encoding='utf-8') as save_except:
			save_except.write(illegal_tokens)
			print(f'Record of illegal tokens saved at: {savepathpobj.joinpath(filenamepobj.stem)}.txt')
	except:
		with open(f'{savepathpobj.joinpath(filenamepobj.stem)}_except.pkl',mode='wb') as save_except_fail_pkl:
			pickle.dump(illegal_tokens,save_except_fail_pkl)
		print(f'Failed saving illegal tokens. PKL saved instead at {savepathpobj.joinpath(filenamepobj.stem)}_except.pkl')
	try:
		with open(f'{savepathpobj.joinpath(filenamepobj.stem)}_stats.txt',mode='w',encoding='utf-8') as save_stats:
			for i in stat:
				save_stats.write(i)
				save_stats.write('\n')
			print(f'Statistics saved at: {savepathpobj.joinpath(filenamepobj.stem)}.txt')
	except:
		print(f'Failed saving statistics.')
	try:
		with open(f'{savepathpobj.joinpath(filenamepobj.stem)}_pdoc.txt',mode='w',encoding='utf-8') as save_pdoc:
			for tkline in proc_lines:
				save_pdoc.write(' '.join(tkline))
				save_pdoc.write('\n')
			print(f'Processed documents saved at: {savepathpobj.joinpath(filenamepobj.stem)}.txt')
	except:

		print(f'Failed saving processed docs.')
	return proc_lines

if __name__ == "__main__":
	language = sys.argv[1]
	source_path = Path(sys.argv[2])
	output_path = Path(sys.argv[3])

	if language == 'english':
		BASIC_LATIN_LOWER = 'abcdefghijklmnopqrstuvwxyz'
		BASIC_LATIN_UPPER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
		BASIC_LATIN_ABC = BASIC_LATIN_LOWER + BASIC_LATIN_UPPER
		LATIN_SUP = 'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ'
		LATIN_EXT_A = 'ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁłŃńŅņŇňŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžſ'
		LATIN_ABC = BASIC_LATIN_ABC + LATIN_SUP + LATIN_EXT_A

		EN_SPECIAL_ABC_LOWER = ''
		EN_SPECIAL_ABC_UPPER = ''
		LANG = 'ENGLISH'
		EN_SPECIAL_ABC = EN_SPECIAL_ABC_LOWER + EN_SPECIAL_ABC_UPPER
		END_PUNCT = '[.!?]'
		GIBBERISH_TABLE = {'тАЩ':'\'','⁠':'','┬а':'','вЂ™':'\'','â':' ','Рађ':'','РђЮ':'','Рђю':'','Рђў':'','РђЎ':'\'','Ã¢â‚¬â„¢':'\'','a¢a,-a"¢':'\'','Ã¢â‚¬Å“':'\'','Ã¢â‚¬Â˜':'\'','Ã¢â‚¬â€':' ','a¯a¿a½--':' ','Ã¢â‚¬â„¢':' ',"Ã¢â‚¬Â¦'":'','Ã¢â‚¬?':'\'','Ã¯Â¿Â½--':'','a¢a,-a"':' ','a¢a,-a':'\'','\u2028':'\n','âÂ€Â™':'\'','\u2008':' ','\u2009':' ','âÂÂ„':'.','Â¡Â®':'\'','âÂ€Â”':' ','Ã¯Â¿':'\'','Ã¯Â¿--':'','Â¡Â¯':'\'','â€™s':'\'','\x84':'\n','\x91':'\'','\x92':'\'','\x93':'\"','\x94':'\"','â€™':'\'','â€':'\'','âÂ€Â‰':'','âÂ€Â˜':'','âÂ€Âœ':'','âÂ€Â':'','â€º':'\'','ÃƒÂ§':'ç','ÃƒÂ¶':'ö','ÃƒÂ±':'ñ','â„¢':'\'','â"¢':'\'','Ã¢â‚¬':'','â€¢':'\'','a€˜':'\'','â€˜':'\'','â€œ':'\'','a€™':'\'','â':'\'','\x07':'·','\x08':'·','\x13':'！','\x15':'⊥','\x1b':'←','\x1d':'↔','\x1e':'△','\x1f':'△','\x80':'€','\x85':'…','\x86':'†','\x87':'‡','\x88':'ˆ','\x89':'‰','\x8a':'Š','\x8c':'Œ','\x95':'·','\x96':'-','\x97':'-','\x98':'~','\x99':'™','\x9a':'š','\x9b':'›','\x9c':'œ','\xa0':' ','\xad':'-',' â‚¬':' ','Titel':'','Ã‚Â…':'','Ã‚Â·':'','ï¿½?':'','Â°-':'-','€šÂ¬':'','â€¡':'a','Â¹':'\'','Ã«':'\'','?Å“':' ','.Å“':' ','â€¦':' ','â€“':'-','â´':'\'','â€”':' ','â€"':' ','a€.':' ','a€¢':' ','a€¦':' ','a¹':'\'','â�?':'','\x19':'','\x1c':'','\x7f':'','\x82':'','\x83':'','â‚¬':' ','Â’':'\'','âÂ€¢':'','Â‘Â‘':'','âÂ€ ':'','âÂ€':' ',' Â‰':' ',' Âœ':' ',' Â™':' ',' Â':' ',' Â˜':' ','â€š':' ','ðŸ˜':'','â€ž':'','â€':'','?':'ß','ã':'ß','Ã¶':'ö','':'\'','Â¢':'¢','Ãœ':'Ü','Ã‰':' ','Ã£':'ã','Ãº':'u','Âºc':'','âºF':'','ÂºF':'','ﬁ':'fi','Ã¤':'ä','Å¡':'.','Ã§':'ç','Â¸':'o','Ã²':'o','Ãª':'e','Ã¼':'ü','Å½':'e','Ã¡':'a','Ã³':'o','ã-':'i','Ã-':'i','ã¨':'a','Ã¨':'a','ã´':'ô','Ã´':'Ô','Â„¢':'','Â"¢':'','â°f':'','Â°F':'','â°c':'','Â°c':'','â°C':'','Â°C':'','Â¤':' ','Ã©':'é','â¡c':'','Â©':'É','ã©':'é','ã±':'ñ','Ã¯':'i','a©':'é','Â¦':' ','â':'','Ã¢':'â',',Â”':'','Â ':' ','�':'\'','ðŸ¤':'','ðÿ¤':'','ðÿ˜':'','Ã®':'î','ã®':'î','ºÂ¬':'','�':'\'','丨':' ','-​':'','':' ','Â£':'','Â¾':'','Â·':'','Ã':'','Â²':'','Ã‚':'','Â¼':'','Â½':'','â®':'','Â®':'','Â-':'','â-':'','Â´':'','':' ','Ã±':'','Â¡':'','':' ','Â¬':'','·':' ','Â•':'','Â¶':'','Â“':'','Â”':'','…':' ','Ã¤':'','':'é','й':'é','Â¸':'','Â­':'','Ã¥':'','Â›':'','ÂŒ':'','Â—':'','Âº':'','Ã¦':'','Â‰':'','Â¿':'','Ã¨':'','Â¯':'','Â†':'','Â¹':'','Ã©':'','Â…':'','Â™':'','':'ž','煤':'ú','聽':' ','帽':'ñ','茅':'é','铆':'í','谩':'á','贸':'ó','﻿':'','​':'','':''}




		DIACRITICS = '̡̢̧̨̛̖̗̘̙̜̝̞̟̠̣̤̥̦̩̪̫̬̭̮̯̰̱̲̳̀́̂̃̄̅̆̇̈̉̊̋̌̍̎̏̐̑̒̓̔̕̚'

		LEGAL_ABC = LATIN_ABC + LATIN_SUP +LATIN_EXT_A + EN_SPECIAL_ABC + DIACRITICS
		LEGAL_DIGIT = '0123456789'
		VOCAL_CHAR = LEGAL_ABC + LEGAL_DIGIT
		LEGAL_PUNCT = string.punctuation + '°' + '$£€¥₹'
		LEGAL_CHAR = LEGAL_ABC + LEGAL_PUNCT + LEGAL_DIGIT

	elif language == 'german':
		BASIC_LATIN_LOWER = 'abcdefghijklmnopqrstuvwxyz'
		BASIC_LATIN_UPPER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
		BASIC_LATIN_ABC = BASIC_LATIN_LOWER + BASIC_LATIN_UPPER
		LATIN_SUP = 'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ'
		LATIN_EXT_A = 'ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁłŃńŅņŇňŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžſ'
		LATIN_ABC = BASIC_LATIN_ABC + LATIN_SUP + LATIN_EXT_A

		DE_SPECIAL_ABC_LOWER = 'äöüß'
		DE_SPECIAL_ABC_UPPER = 'ÄÖÜẞ'
		LANG = 'GERMAN'
		DE_SPECIAL_ABC = DE_SPECIAL_ABC_LOWER + DE_SPECIAL_ABC_UPPER
		END_PUNCT = '[.!?]'
		GIBBERISH_TABLE = {'тАЩ':'\'','⁠':'','┬а':'','вЂ™':'\'','â':' ','Рађ':'','РђЮ':'','Рђю':'','Рђў':'','РђЎ':'\'','Ã¢â‚¬â„¢':'\'','a¢a,-a"¢':'\'','Ã¢â‚¬Å“':'\'','Ã¢â‚¬Â˜':'\'','Ã¢â‚¬â€':' ','a¯a¿a½--':' ','Ã¢â‚¬â„¢':' ',"Ã¢â‚¬Â¦'":'','Ã¢â‚¬?':'\'','Ã¯Â¿Â½--':'','a¢a,-a"':' ','a¢a,-a':'\'','\u2028':'\n','âÂ€Â™':'\'','\u2008':' ','\u2009':' ','âÂÂ„':'.','Â¡Â®':'\'','âÂ€Â”':' ','Ã¯Â¿':'\'','Ã¯Â¿--':'','Â¡Â¯':'\'','â€™s':'\'','\x84':'\n','\x91':'\'','\x92':'\'','\x93':'\"','\x94':'\"','â€™':'\'','â€':'\'','âÂ€Â‰':'','âÂ€Â˜':'','âÂ€Âœ':'','âÂ€Â':'','â€º':'\'','ÃƒÂ§':'ç','ÃƒÂ¶':'ö','ÃƒÂ±':'ñ','â„¢':'\'','â"¢':'\'','Ã¢â‚¬':'','â€¢':'\'','a€˜':'\'','â€˜':'\'','â€œ':'\'','a€™':'\'','â':'\'','\x07':'·','\x08':'·','\x13':'！','\x15':'⊥','\x1b':'←','\x1d':'↔','\x1e':'△','\x1f':'△','\x80':'€','\x85':'…','\x86':'†','\x87':'‡','\x88':'ˆ','\x89':'‰','\x8a':'Š','\x8c':'Œ','\x95':'·','\x96':'-','\x97':'-','\x98':'~','\x99':'™','\x9a':'š','\x9b':'›','\x9c':'œ','\xa0':' ','\xad':'-',' â‚¬':' ','Titel':'','Ã‚Â…':'','Ã‚Â·':'','ï¿½?':'','Â°-':'-','€šÂ¬':'','â€¡':'a','Â¹':'\'','Ã«':'\'','?Å“':' ','.Å“':' ','â€¦':' ','â€“':'-','â´':'\'','â€”':' ','â€"':' ','a€.':' ','a€¢':' ','a€¦':' ','a¹':'\'','â�?':'','\x19':'','\x1c':'','\x7f':'','\x82':'','\x83':'','â‚¬':' ','Â’':'\'','âÂ€¢':'','Â‘Â‘':'','âÂ€ ':'','âÂ€':' ',' Â‰':' ',' Âœ':' ',' Â™':' ',' Â':' ',' Â˜':' ','â€š':' ','ðŸ˜':'','â€ž':'','â€':'','?':'ß','ã':'ß','Ã¶':'ö','':'\'','Â¢':'¢','Ãœ':'Ü','Ã‰':' ','Ã£':'ã','Ãº':'u','Âºc':'','âºF':'','ÂºF':'','ﬁ':'fi','Ã¤':'ä','Å¡':'.','Ã§':'ç','Â¸':'o','Ã²':'o','Ãª':'e','Ã¼':'ü','Å½':'e','Ã¡':'a','Ã³':'o','ã-':'i','Ã-':'i','ã¨':'a','Ã¨':'a','ã´':'ô','Ã´':'Ô','Â„¢':'','Â"¢':'','â°f':'','Â°F':'','â°c':'','Â°c':'','â°C':'','Â°C':'','Â¤':' ','Ã©':'é','â¡c':'','Â©':'É','ã©':'é','ã±':'ñ','Ã¯':'i','a©':'é','Â¦':' ','â':'','Ã¢':'â',',Â”':'','Â ':' ','�':'\'','ðŸ¤':'','ðÿ¤':'','ðÿ˜':'','Ã®':'î','ã®':'î','ºÂ¬':'','�':'\'','丨':' ','-​':'','':' ','Â£':'','Â¾':'','Â·':'','Ã':'','Â²':'','Ã‚':'','Â¼':'','Â½':'','â®':'','Â®':'','Â-':'','â-':'','Â´':'','':' ','Ã±':'','Â¡':'','':' ','Â¬':'','·':' ','Â•':'','Â¶':'','Â“':'','Â”':'','…':' ','Ã¤':'','':'é','й':'é','Â¸':'','Â­':'','Ã¥':'','Â›':'','ÂŒ':'','Â—':'','Âº':'','Ã¦':'','Â‰':'','Â¿':'','Ã¨':'','Â¯':'','Â†':'','Â¹':'','Ã©':'','Â…':'','Â™':'','':'ž','煤':'ú','聽':' ','帽':'ñ','茅':'é','铆':'í','谩':'á','贸':'ó','﻿':'','​':'','':''}




		DIACRITICS = '̡̢̧̨̛̖̗̘̙̜̝̞̟̠̣̤̥̦̩̪̫̬̭̮̯̰̱̲̳̀́̂̃̄̅̆̇̈̉̊̋̌̍̎̏̐̑̒̓̔̕̚'

		LEGAL_ABC = LATIN_ABC + LATIN_SUP +LATIN_EXT_A + DE_SPECIAL_ABC + DIACRITICS
		LEGAL_DIGIT = '0123456789'
		VOCAL_CHAR = LEGAL_ABC + LEGAL_DIGIT
		LEGAL_PUNCT = string.punctuation + '°' + '$£€¥₹'
		LEGAL_CHAR = LEGAL_ABC + LEGAL_PUNCT + LEGAL_DIGIT
	elif language == 'french':
		BASIC_LATIN_LOWER = 'abcdefghijklmnopqrstuvwxyz'
		BASIC_LATIN_UPPER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
		BASIC_LATIN_ABC = BASIC_LATIN_LOWER + BASIC_LATIN_UPPER
		LATIN_SUP = 'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ'
		LATIN_EXT_A = 'ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁłŃńŅņŇňŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžſ'
		LATIN_ABC = BASIC_LATIN_ABC + LATIN_SUP + LATIN_EXT_A

		FR_SPECIAL_ABC_LOWER = 'àâèéëêïîôœæùûüçÿ'
		FR_SPECIAL_ABC_UPPER = 'ÀÂÈÉËÊÏÎÔŒÆÙÛÜÇŸ'
		LANG = 'FRENCH'
		FR_SPECIAL_ABC = FR_SPECIAL_ABC_LOWER + FR_SPECIAL_ABC_UPPER
		END_PUNCT = '[.!?]'
		GIBBERISH_TABLE = {'тАЩ':'\'','⁠':'','┬а':'','вЂ™':'\'','â':' ','Рађ':'','РђЮ':'','Рђю':'','Рђў':'','РђЎ':'\'','Ã¢â‚¬â„¢':'\'','a¢a,-a"¢':'\'','Ã¢â‚¬Å“':'\'','Ã¢â‚¬Â˜':'\'','Ã¢â‚¬â€':' ','a¯a¿a½--':' ','Ã¢â‚¬â„¢':' ',"Ã¢â‚¬Â¦'":'','Ã¢â‚¬?':'\'','Ã¯Â¿Â½--':'','a¢a,-a"':' ','a¢a,-a':'\'','\u2028':'\n','âÂ€Â™':'\'','\u2008':' ','\u2009':' ','âÂÂ„':'.','Â¡Â®':'\'','âÂ€Â”':' ','Ã¯Â¿':'\'','Ã¯Â¿--':'','Â¡Â¯':'\'','â€™s':'\'','\x84':'\n','\x91':'\'','\x92':'\'','\x93':'\"','\x94':'\"','â€™':'\'','â€':'\'','âÂ€Â‰':'','âÂ€Â˜':'','âÂ€Âœ':'','âÂ€Â':'','â€º':'\'','ÃƒÂ§':'ç','ÃƒÂ¶':'ö','ÃƒÂ±':'ñ','â„¢':'\'','â"¢':'\'','Ã¢â‚¬':'','â€¢':'\'','a€˜':'\'','â€˜':'\'','â€œ':'\'','a€™':'\'','â':'\'','\x07':'·','\x08':'·','\x13':'！','\x15':'⊥','\x1b':'←','\x1d':'↔','\x1e':'△','\x1f':'△','\x80':'€','\x85':'…','\x86':'†','\x87':'‡','\x88':'ˆ','\x89':'‰','\x8a':'Š','\x8c':'Œ','\x95':'·','\x96':'-','\x97':'-','\x98':'~','\x99':'™','\x9a':'š','\x9b':'›','\x9c':'œ','\xa0':' ','\xad':'-',' â‚¬':' ','Titel':'','Ã‚Â…':'','Ã‚Â·':'','ï¿½?':'','Â°-':'-','€šÂ¬':'','â€¡':'a','Â¹':'\'','Ã«':'\'','?Å“':' ','.Å“':' ','â€¦':' ','â€“':'-','â´':'\'','â€”':' ','â€"':' ','a€.':' ','a€¢':' ','a€¦':' ','a¹':'\'','â�?':'','\x19':'','\x1c':'','\x7f':'','\x82':'','\x83':'','â‚¬':' ','Â’':'\'','âÂ€¢':'','Â‘Â‘':'','âÂ€ ':'','âÂ€':' ',' Â‰':' ',' Âœ':' ',' Â™':' ',' Â':' ',' Â˜':' ','â€š':' ','ðŸ˜':'','â€ž':'','â€':'','?':'ß','ã':'ß','Ã¶':'ö','':'\'','Â¢':'¢','Ãœ':'Ü','Ã‰':' ','Ã£':'ã','Ãº':'u','Âºc':'','âºF':'','ÂºF':'','ﬁ':'fi','Ã¤':'ä','Å¡':'.','Ã§':'ç','Â¸':'o','Ã²':'o','Ãª':'e','Ã¼':'ü','Å½':'e','Ã¡':'a','Ã³':'o','ã-':'i','Ã-':'i','ã¨':'a','Ã¨':'a','ã´':'ô','Ã´':'Ô','Â„¢':'','Â"¢':'','â°f':'','Â°F':'','â°c':'','Â°c':'','â°C':'','Â°C':'','Â¤':' ','Ã©':'é','â¡c':'','Â©':'É','ã©':'é','ã±':'ñ','Ã¯':'i','a©':'é','Â¦':' ','â':'','Ã¢':'â',',Â”':'','Â ':' ','�':'\'','ðŸ¤':'','ðÿ¤':'','ðÿ˜':'','Ã®':'î','ã®':'î','ºÂ¬':'','�':'\'','丨':' ','-​':'','':' ','Â£':'','Â¾':'','Â·':'','Ã':'','Â²':'','Ã‚':'','Â¼':'','Â½':'','â®':'','Â®':'','Â-':'','â-':'','Â´':'','':' ','Ã±':'','Â¡':'','':' ','Â¬':'','·':' ','Â•':'','Â¶':'','Â“':'','Â”':'','…':' ','Ã¤':'','':'é','й':'é','Â¸':'','Â­':'','Ã¥':'','Â›':'','ÂŒ':'','Â—':'','Âº':'','Ã¦':'','Â‰':'','Â¿':'','Ã¨':'','Â¯':'','Â†':'','Â¹':'','Ã©':'','Â…':'','Â™':'','':'ž','煤':'ú','聽':' ','帽':'ñ','茅':'é','铆':'í','谩':'á','贸':'ó','﻿':'','​':'','':''}




		DIACRITICS = '̡̢̧̨̛̖̗̘̙̜̝̞̟̠̣̤̥̦̩̪̫̬̭̮̯̰̱̲̳̀́̂̃̄̅̆̇̈̉̊̋̌̍̎̏̐̑̒̓̔̕̚'

		LEGAL_ABC = LATIN_ABC + LATIN_SUP +LATIN_EXT_A + FR_SPECIAL_ABC + DIACRITICS
		LEGAL_DIGIT = '0123456789'
		VOCAL_CHAR = LEGAL_ABC + LEGAL_DIGIT
		LEGAL_PUNCT = string.punctuation + '°' + '$£€¥₹'
		LEGAL_CHAR = LEGAL_ABC + LEGAL_PUNCT + LEGAL_DIGIT
	elif language == 'spanish':
		BASIC_LATIN_LOWER = 'abcdefghijklmnopqrstuvwxyz'
		BASIC_LATIN_UPPER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
		BASIC_LATIN_ABC = BASIC_LATIN_LOWER + BASIC_LATIN_UPPER
		LATIN_SUP = 'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ'
		LATIN_EXT_A = 'ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁłŃńŅņŇňŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžſ'
		LATIN_ABC = BASIC_LATIN_ABC + LATIN_SUP + LATIN_EXT_A

		ES_SPECIAL_ABC_LOWER = 'áéíóúüñïý'
		ES_SPECIAL_ABC_UPPER = 'ÁÉÍÓÚÜÑÏÝ'
		LANG = 'SPANISH'
		ES_SPECIAL_ABC = ES_SPECIAL_ABC_LOWER + ES_SPECIAL_ABC_UPPER
		END_PUNCT = '[.!?¿¡]'
		GIBBERISH_TABLE = {'тАЩ':'\'','⁠':'','┬а':'','вЂ™':'\'','â':' ','Рађ':'','РђЮ':'','Рђю':'','Рђў':'','РђЎ':'\'','Ã¢â‚¬â„¢':'\'','a¢a,-a"¢':'\'','Ã¢â‚¬Å“':'\'','Ã¢â‚¬Â˜':'\'','Ã¢â‚¬â€':' ','a¯a¿a½--':' ','Ã¢â‚¬â„¢':' ',"Ã¢â‚¬Â¦'":'','Ã¢â‚¬?':'\'','Ã¯Â¿Â½--':'','a¢a,-a"':' ','a¢a,-a':'\'','\u2028':'\n','âÂ€Â™':'\'','\u2008':' ','\u2009':' ','âÂÂ„':'.','Â¡Â®':'\'','âÂ€Â”':' ','Ã¯Â¿':'\'','Ã¯Â¿--':'','Â¡Â¯':'\'','â€™s':'\'','\x84':'\n','\x91':'\'','\x92':'\'','\x93':'\"','\x94':'\"','â€™':'\'','â€':'\'','âÂ€Â‰':'','âÂ€Â˜':'','âÂ€Âœ':'','âÂ€Â':'','â€º':'\'','ÃƒÂ§':'ç','ÃƒÂ¶':'ö','ÃƒÂ±':'ñ','â„¢':'\'','â"¢':'\'','Ã¢â‚¬':'','â€¢':'\'','a€˜':'\'','â€˜':'\'','â€œ':'\'','a€™':'\'','â':'\'','\x07':'·','\x08':'·','\x13':'！','\x15':'⊥','\x1b':'←','\x1d':'↔','\x1e':'△','\x1f':'△','\x80':'€','\x85':'…','\x86':'†','\x87':'‡','\x88':'ˆ','\x89':'‰','\x8a':'Š','\x8c':'Œ','\x95':'·','\x96':'-','\x97':'-','\x98':'~','\x99':'™','\x9a':'š','\x9b':'›','\x9c':'œ','\xa0':' ','\xad':'-',' â‚¬':' ','Titel':'','Ã‚Â…':'','Ã‚Â·':'','ï¿½?':'','Â°-':'-','€šÂ¬':'','â€¡':'a','Â¹':'\'','Ã«':'\'','?Å“':' ','.Å“':' ','â€¦':' ','â€“':'-','â´':'\'','â€”':' ','â€"':' ','a€.':' ','a€¢':' ','a€¦':' ','a¹':'\'','â�?':'','\x19':'','\x1c':'','\x7f':'','\x82':'','\x83':'','â‚¬':' ','Â’':'\'','âÂ€¢':'','Â‘Â‘':'','âÂ€ ':'','âÂ€':' ',' Â‰':' ',' Âœ':' ',' Â™':' ',' Â':' ',' Â˜':' ','â€š':' ','ðŸ˜':'','â€ž':'','â€':'','?':'ß','ã':'ß','Ã¶':'ö','':'\'','Â¢':'¢','Ãœ':'Ü','Ã‰':' ','Ã£':'ã','Ãº':'u','Âºc':'','âºF':'','ÂºF':'','ﬁ':'fi','Ã¤':'ä','Å¡':'.','Ã§':'ç','Â¸':'o','Ã²':'o','Ãª':'e','Ã¼':'ü','Å½':'e','Ã¡':'a','Ã³':'o','ã-':'i','Ã-':'i','ã¨':'a','Ã¨':'a','ã´':'ô','Ã´':'Ô','Â„¢':'','Â"¢':'','â°f':'','Â°F':'','â°c':'','Â°c':'','â°C':'','Â°C':'','Â¤':' ','Ã©':'é','â¡c':'','Â©':'É','ã©':'é','ã±':'ñ','Ã¯':'i','a©':'é','Â¦':' ','â':'','Ã¢':'â',',Â”':'','Â ':' ','�':'\'','ðŸ¤':'','ðÿ¤':'','ðÿ˜':'','Ã®':'î','ã®':'î','ºÂ¬':'','�':'\'','丨':' ','-​':'','':' ','Â£':'','Â¾':'','Â·':'','Ã':'','Â²':'','Ã‚':'','Â¼':'','Â½':'','â®':'','Â®':'','Â-':'','â-':'','Â´':'','':' ','Ã±':'','Â¡':'','':' ','Â¬':'','·':' ','Â•':'','Â¶':'','Â“':'','Â”':'','…':' ','Ã¤':'','':'é','й':'é','Â¸':'','Â­':'','Ã¥':'','Â›':'','ÂŒ':'','Â—':'','Âº':'','Ã¦':'','Â‰':'','Â¿':'','Ã¨':'','Â¯':'','Â†':'','Â¹':'','Ã©':'','Â…':'','Â™':'','':'ž','煤':'ú','聽':' ','帽':'ñ','茅':'é','铆':'í','谩':'á','贸':'ó','﻿':'','​':'','':''}




		DIACRITICS = '̡̢̧̨̛̖̗̘̙̜̝̞̟̠̣̤̥̦̩̪̫̬̭̮̯̰̱̲̳̀́̂̃̄̅̆̇̈̉̊̋̌̍̎̏̐̑̒̓̔̕̚'

		LEGAL_ABC = LATIN_ABC + LATIN_SUP +LATIN_EXT_A + ES_SPECIAL_ABC + DIACRITICS
		LEGAL_DIGIT = '0123456789'
		VOCAL_CHAR = LEGAL_ABC + LEGAL_DIGIT
		LEGAL_PUNCT = string.punctuation + '°' + '$£€¥₹' + '¿¡'
		LEGAL_CHAR = LEGAL_ABC + LEGAL_PUNCT + LEGAL_DIGIT
	elif language == 'russian':
		BASIC_LATIN_LOWER = 'abcdefghijklmnopqrstuvwxyz'
		BASIC_LATIN_UPPER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
		BASIC_LATIN_ABC = BASIC_LATIN_LOWER + BASIC_LATIN_UPPER
		LATIN_SUP = 'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ'
		LATIN_EXT_A = 'ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁłŃńŅņŇňŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžſ'
		LATIN_ABC = BASIC_LATIN_ABC + LATIN_SUP + LATIN_EXT_A

		RU_SPECIAL_ABC_LOWER = 'абвгдеёіїжзийклмнопрстуфхцчшщъыьэюя'
		RU_SPECIAL_ABC_UPPER = 'АБВГДЕЁІЇЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
		LANG = 'RUSSIAN'
		RU_SPECIAL_ABC = RU_SPECIAL_ABC_LOWER + RU_SPECIAL_ABC_UPPER
		END_PUNCT = '[.!?]'
		GIBBERISH_TABLE = {'тАЩ':'\'','⁠':'','┬а':'','вЂ™':'\'','â':' ','Рађ':'','РђЮ':'','Рђю':'','Рђў':'','РђЎ':'\'','Ã¢â‚¬â„¢':'\'','a¢a,-a"¢':'\'','Ã¢â‚¬Å“':'\'','Ã¢â‚¬Â˜':'\'','Ã¢â‚¬â€':' ','a¯a¿a½--':' ','Ã¢â‚¬â„¢':' ',"Ã¢â‚¬Â¦'":'','Ã¢â‚¬?':'\'','Ã¯Â¿Â½--':'','a¢a,-a"':' ','a¢a,-a':'\'','\u2028':'\n','âÂ€Â™':'\'','\u2008':' ','\u2009':' ','âÂÂ„':'.','Â¡Â®':'\'','âÂ€Â”':' ','Ã¯Â¿':'\'','Ã¯Â¿--':'','Â¡Â¯':'\'','â€™s':'\'','\x84':'\n','\x91':'\'','\x92':'\'','\x93':'\"','\x94':'\"','â€™':'\'','â€':'\'','âÂ€Â‰':'','âÂ€Â˜':'','âÂ€Âœ':'','âÂ€Â':'','â€º':'\'','ÃƒÂ§':'ç','ÃƒÂ¶':'ö','ÃƒÂ±':'ñ','â„¢':'\'','â"¢':'\'','Ã¢â‚¬':'','â€¢':'\'','a€˜':'\'','â€˜':'\'','â€œ':'\'','a€™':'\'','â':'\'','\x07':'·','\x08':'·','\x13':'！','\x15':'⊥','\x1b':'←','\x1d':'↔','\x1e':'△','\x1f':'△','\x80':'€','\x85':'…','\x86':'†','\x87':'‡','\x88':'ˆ','\x89':'‰','\x8a':'Š','\x8c':'Œ','\x95':'·','\x96':'-','\x97':'-','\x98':'~','\x99':'™','\x9a':'š','\x9b':'›','\x9c':'œ','\xa0':' ','\xad':'-',' â‚¬':' ','Titel':'','Ã‚Â…':'','Ã‚Â·':'','ï¿½?':'','Â°-':'-','€šÂ¬':'','â€¡':'a','Â¹':'\'','Ã«':'\'','?Å“':' ','.Å“':' ','â€¦':' ','â€“':'-','â´':'\'','â€”':' ','â€"':' ','a€.':' ','a€¢':' ','a€¦':' ','a¹':'\'','â�?':'','\x19':'','\x1c':'','\x7f':'','\x82':'','\x83':'','â‚¬':' ','Â’':'\'','âÂ€¢':'','Â‘Â‘':'','âÂ€ ':'','âÂ€':' ',' Â‰':' ',' Âœ':' ',' Â™':' ',' Â':' ',' Â˜':' ','â€š':' ','ðŸ˜':'','â€ž':'','â€':'','?':'ß','ã':'ß','Ã¶':'ö','':'\'','Â¢':'¢','Ãœ':'Ü','Ã‰':' ','Ã£':'ã','Ãº':'u','Âºc':'','âºF':'','ÂºF':'','ﬁ':'fi','Ã¤':'ä','Å¡':'.','Ã§':'ç','Â¸':'o','Ã²':'o','Ãª':'e','Ã¼':'ü','Å½':'e','Ã¡':'a','Ã³':'o','ã-':'i','Ã-':'i','ã¨':'a','Ã¨':'a','ã´':'ô','Ã´':'Ô','Â„¢':'','Â"¢':'','â°f':'','Â°F':'','â°c':'','Â°c':'','â°C':'','Â°C':'','Â¤':' ','Ã©':'é','â¡c':'','Â©':'É','ã©':'é','ã±':'ñ','Ã¯':'i','a©':'é','Â¦':' ','â':'','Ã¢':'â',',Â”':'','Â ':' ','�':'\'','ðŸ¤':'','ðÿ¤':'','ðÿ˜':'','Ã®':'î','ã®':'î','ºÂ¬':'','�':'\'','丨':' ','-​':'','':' ','Â£':'','Â¾':'','Â·':'','Ã':'','Â²':'','Ã‚':'','Â¼':'','Â½':'','â®':'','Â®':'','Â-':'','â-':'','Â´':'','':' ','Ã±':'','Â¡':'','':' ','Â¬':'','·':' ','Â•':'','Â¶':'','Â“':'','Â”':'','…':' ','Ã¤':'','':'é','й':'é','Â¸':'','Â­':'','Ã¥':'','Â›':'','ÂŒ':'','Â—':'','Âº':'','Ã¦':'','Â‰':'','Â¿':'','Ã¨':'','Â¯':'','Â†':'','Â¹':'','Ã©':'','Â…':'','Â™':'','':'ž','煤':'ú','聽':' ','帽':'ñ','茅':'é','铆':'í','谩':'á','贸':'ó','﻿':'','​':'','':''}




		DIACRITICS = '̡̢̧̨̛̖̗̘̙̜̝̞̟̠̣̤̥̦̩̪̫̬̭̮̯̰̱̲̳̀́̂̃̄̅̆̇̈̉̊̋̌̍̎̏̐̑̒̓̔̕̚'

		LEGAL_ABC = LATIN_ABC + LATIN_SUP +LATIN_EXT_A + RU_SPECIAL_ABC + DIACRITICS
		LEGAL_DIGIT = '0123456789'
		VOCAL_CHAR = LEGAL_ABC + LEGAL_DIGIT
		LEGAL_PUNCT = string.punctuation + '°' + '$£€¥₹'
		LEGAL_CHAR = LEGAL_ABC + LEGAL_PUNCT + LEGAL_DIGIT
	elif language == 'czech':
		BASIC_LATIN_LOWER = 'abcdefghijklmnopqrstuvwxyz'
		BASIC_LATIN_UPPER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
		BASIC_LATIN_ABC = BASIC_LATIN_LOWER + BASIC_LATIN_UPPER
		LATIN_SUP = 'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ'
		LATIN_EXT_A = 'ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁłŃńŅņŇňŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžſ'
		LATIN_ABC = BASIC_LATIN_ABC + LATIN_SUP + LATIN_EXT_A

		CZ_SPECIAL_ABC_LOWER = 'áéěíóúůýčďňřšťž'
		CZ_SPECIAL_ABC_UPPER = 'ÁÉĚÍÓÚŮÝČĎŇŘŠŤŽ'
		LANG = 'CZECH'
		CZ_SPECIAL_ABC = CZ_SPECIAL_ABC_LOWER + CZ_SPECIAL_ABC_UPPER
		END_PUNCT = '[.!?]'
		GIBBERISH_TABLE = {'тАЩ':'\'','⁠':'','┬а':'','вЂ™':'\'','â':' ','Рађ':'','РђЮ':'','Рђю':'','Рђў':'','РђЎ':'\'','Ã¢â‚¬â„¢':'\'','a¢a,-a"¢':'\'','Ã¢â‚¬Å“':'\'','Ã¢â‚¬Â˜':'\'','Ã¢â‚¬â€':' ','a¯a¿a½--':' ','Ã¢â‚¬â„¢':' ',"Ã¢â‚¬Â¦'":'','Ã¢â‚¬?':'\'','Ã¯Â¿Â½--':'','a¢a,-a"':' ','a¢a,-a':'\'','\u2028':'\n','âÂ€Â™':'\'','\u2008':' ','\u2009':' ','âÂÂ„':'.','Â¡Â®':'\'','âÂ€Â”':' ','Ã¯Â¿':'\'','Ã¯Â¿--':'','Â¡Â¯':'\'','â€™s':'\'','\x84':'\n','\x91':'\'','\x92':'\'','\x93':'\"','\x94':'\"','â€™':'\'','â€':'\'','âÂ€Â‰':'','âÂ€Â˜':'','âÂ€Âœ':'','âÂ€Â':'','â€º':'\'','ÃƒÂ§':'ç','ÃƒÂ¶':'ö','ÃƒÂ±':'ñ','â„¢':'\'','â"¢':'\'','Ã¢â‚¬':'','â€¢':'\'','a€˜':'\'','â€˜':'\'','â€œ':'\'','a€™':'\'','â':'\'','\x07':'·','\x08':'·','\x13':'！','\x15':'⊥','\x1b':'←','\x1d':'↔','\x1e':'△','\x1f':'△','\x80':'€','\x85':'…','\x86':'†','\x87':'‡','\x88':'ˆ','\x89':'‰','\x8a':'Š','\x8c':'Œ','\x95':'·','\x96':'-','\x97':'-','\x98':'~','\x99':'™','\x9a':'š','\x9b':'›','\x9c':'œ','\xa0':' ','\xad':'-',' â‚¬':' ','Titel':'','Ã‚Â…':'','Ã‚Â·':'','ï¿½?':'','Â°-':'-','€šÂ¬':'','â€¡':'a','Â¹':'\'','Ã«':'\'','?Å“':' ','.Å“':' ','â€¦':' ','â€“':'-','â´':'\'','â€”':' ','â€"':' ','a€.':' ','a€¢':' ','a€¦':' ','a¹':'\'','â�?':'','\x19':'','\x1c':'','\x7f':'','\x82':'','\x83':'','â‚¬':' ','Â’':'\'','âÂ€¢':'','Â‘Â‘':'','âÂ€ ':'','âÂ€':' ',' Â‰':' ',' Âœ':' ',' Â™':' ',' Â':' ',' Â˜':' ','â€š':' ','ðŸ˜':'','â€ž':'','â€':'','?':'ß','ã':'ß','Ã¶':'ö','':'\'','Â¢':'¢','Ãœ':'Ü','Ã‰':' ','Ã£':'ã','Ãº':'u','Âºc':'','âºF':'','ÂºF':'','ﬁ':'fi','Ã¤':'ä','Å¡':'.','Ã§':'ç','Â¸':'o','Ã²':'o','Ãª':'e','Ã¼':'ü','Å½':'e','Ã¡':'a','Ã³':'o','ã-':'i','Ã-':'i','ã¨':'a','Ã¨':'a','ã´':'ô','Ã´':'Ô','Â„¢':'','Â"¢':'','â°f':'','Â°F':'','â°c':'','Â°c':'','â°C':'','Â°C':'','Â¤':' ','Ã©':'é','â¡c':'','Â©':'É','ã©':'é','ã±':'ñ','Ã¯':'i','a©':'é','Â¦':' ','â':'','Ã¢':'â',',Â”':'','Â ':' ','�':'\'','ðŸ¤':'','ðÿ¤':'','ðÿ˜':'','Ã®':'î','ã®':'î','ºÂ¬':'','�':'\'','丨':' ','-​':'','':' ','Â£':'','Â¾':'','Â·':'','Ã':'','Â²':'','Ã‚':'','Â¼':'','Â½':'','â®':'','Â®':'','Â-':'','â-':'','Â´':'','':' ','Ã±':'','Â¡':'','':' ','Â¬':'','·':' ','Â•':'','Â¶':'','Â“':'','Â”':'','…':' ','Ã¤':'','':'é','й':'é','Â¸':'','Â­':'','Ã¥':'','Â›':'','ÂŒ':'','Â—':'','Âº':'','Ã¦':'','Â‰':'','Â¿':'','Ã¨':'','Â¯':'','Â†':'','Â¹':'','Ã©':'','Â…':'','Â™':'','':'ž','煤':'ú','聽':' ','帽':'ñ','茅':'é','铆':'í','谩':'á','贸':'ó','﻿':'','​':'','':''}




		DIACRITICS = '̡̢̧̨̛̖̗̘̙̜̝̞̟̠̣̤̥̦̩̪̫̬̭̮̯̰̱̲̳̀́̂̃̄̅̆̇̈̉̊̋̌̍̎̏̐̑̒̓̔̕̚'

		LEGAL_ABC = LATIN_ABC + LATIN_SUP +LATIN_EXT_A + CZ_SPECIAL_ABC + DIACRITICS
		LEGAL_DIGIT = '0123456789'
		VOCAL_CHAR = LEGAL_ABC + LEGAL_DIGIT
		LEGAL_PUNCT = string.punctuation + '°' + '$£€¥₹'
		LEGAL_CHAR = LEGAL_ABC + LEGAL_PUNCT + LEGAL_DIGIT
	else:
		raise ValueError('unsupported language')

	for txt in os.listdir(source_path):
		pipeline(f'{source_path.joinpath(txt)}',output_path)
