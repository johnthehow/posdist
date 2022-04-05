# Linear-position-distribution Analysis
This repository contains code for 'Emergent linguistic regularities learned by neural network language models via linear position distribution'.
It includes code for 功能1, 功能2(对应文中位置), ...

## Requirements
For pre-processing Leipzig news corpora:
* [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)

Statistics and visualizations of Linear-position-distribution:
* [Numpy](http://www.numpy.org/)
* [SciPy](https://scipy.org/)
* [MatplotLib](https://matplotlib.org/)

Extracting attention weights for words in sentences:
* [PyTorch](https://pytorch.org/)
* [Huggingface Transformers](https://huggingface.co/)



## Pre-processing Leipzig news corpora
We provide a script `corpus_prep.py` for pre-processing Leipzig news corpora. This script is capable of cleaning news texts in six languages (English, German, French, Spanish, Russian, Czech) from [Leipzig Corpora Collection](https://corpora.uni-leipzig.de/).

Corpora used in the paper can be downloaded from <https://wortschatz.uni-leipzig.de/en/download/>. Some neccessay steps are required before runing the pre-processing script: 

1. Extract plain-text corpora (`<language>_news_<year>_1M-sentences.txt`) from compressed archives (`<language>_news_<year>_1M.tar.gz`). 
2. Put extracted text files under same directory for further reference. 

To get a corpus covering all years of collection, simply concatenate all extracted texts.

Corpus pre-preprocessing is run with:

`python corpus_prep.py <language> <path-to-your-data> <output-path>`

Sample usage:

`python corpus_prep.py english ~/source ~/output`

Output:

`<language>_news_<year>_1M-sentences.pkl`: pre-processed text (Python object serialization)

`<language>_news_<year>_1M-sentences_except.txt`: a list of illegal tokens excluded from pre-processed text

`<language>_news_<year>_1M-sentences_pdoc.txt`: pre-processed text

`<language>_news_<year>_1M-sentences_stats.txt`: a summary of the pre-processed text



## Statistics and visualizations of linear-position-distribution

## Extracting attention weights for words in sentences

