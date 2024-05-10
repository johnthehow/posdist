'''
[id] 20240506202850
[desc] 1) 使用spacy进行中文分词和词性标注
[dep]
    1) spacy
    2) spacy.zh_core_web_sm
[usage]
    1) standalone (ifname=main)
    2) acceleratable (feeder_processor)
'''

from pathlib import Path
import spacy

def main(filename, lang):
    if lang == 'zh':
        nlp = spacy.load('zh_core_web_sm')
    elif lang == 'jp':
        nlp = spacy.load('ja_core_news_sm')
    elif lang == 'kr':
        nlp = spacy.load('ko_core_news_sm')
    else:
        raise ValueError(f'{lang} is not supported')
    filename = Path(filename)
    output_filename = filename.parent.joinpath(f'{filename.name}.tokenpos')
    output_io = open(output_filename, mode='w', encoding='utf-8')
    with open(filename, mode='r', encoding='utf-8') as file:
        doc = nlp.pipe(file, disable = ['ner','lemmatizer','parser']) # disable这些组件能提速约40%
        for sent in doc:
            output_io.write(f' '.join([f'{token.text}:{token.pos_}' for token in sent if token.text != '\n']))
            output_io.write('\n')
    output_io.close()
    return