# Linear-position-distribution Analysis
This repository contains code for 'Emergent linguistic regularities learned by neural network language models via linear position distribution'.
It includes code for 功能1, 功能2(对应文中位置), ...

## Requirements
All scirpts in this repo require Python 3.6+

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
We provide a script `corpus_prep.py` for pre-processing Leipzig news corpora. This script is capable of cleaning news texts in six languages (English, German, French, Spanish, Russian and Czech) from [Leipzig Corpora Collection](https://corpora.uni-leipzig.de/).

Corpora used in the paper can be downloaded from <https://wortschatz.uni-leipzig.de/en/download/>. Some neccessay steps are required before runing the pre-processing script: 

1. Extract plain-text corpora (`<language>_news_<year>_1M-sentences.txt`) from compressed archives (`<language>_news_<year>_1M.tar.gz`). 
2. Put extracted text files under a same directory for further reference. 

To get a corpus covering all years of collection, simply concatenate all extracted texts.

Corpus pre-preprocessing is run with:

`python corpus_prep.py <language> <path-to-your-data> <output-path>`

Sample usage:

`python corpus_prep.py english ~/source ~/output`

For each plain-text corpus text, following output files are produced by this script:

* `<language>_news_<year>_1M-sentences.pkl` pre-processed text (Python object serialization)

* `<language>_news_<year>_1M-sentences_except.txt` a list of illegal tokens excluded from pre-processed text

* `<language>_news_<year>_1M-sentences_pdoc.txt` pre-processed text

* `<language>_news_<year>_1M-sentences_stats.txt` a summary of the pre-processed text

Only files named `<language>_news_<year>_1M-sentences.pkl` are used for further analysis.

## Statistics and visualizations of linear-position-distribution
### Descriptive statistics (see Table 2 of the paper)
![Mean D_{KL_{lengths}}](https://latex.codecogs.com/svg.image?Mean%20D_%7BKL_%7Blengths%7D%7D) is computed by `posdist_stat_lengths.py`

Usage:

`python posdist_stat_lengths.py <start-sentence-length> <end-sentence-length> <KDE-bandwidth> <corpus-dir> <word-list>`

![Mean D_{KL_{words}}](https://latex.codecogs.com/svg.image?Mean%20D_%7BKL_%7Bwords%7D%7D) is computed by `posdist_stat_words.py`

Usage:

`python posdist_stat_words.py  <start-sentence-length> <end-sentence-length> <KDE-bandwidth> <corpus-dir> <word-list>`


![Mean D_{KL_{years}}](https://latex.codecogs.com/svg.image?Mean%20D_%7BKL_%7Byears%7D%7D) is computed by `posdist_stat_years.py`

Usage:

`python posdist_stat_years.py  <start-sentence-length> <end-sentence-length> <KDE-bandwidth> <corpus-dir> <word-list>`

### Visualizations
`posdist_viz_lengths.py` produces the figure showing "pattern consistency over sentence lengths" for linear-position-distributions of function words. Follow the prompts to get the figure (as Fig. 1. of the paper).

`posdist_viz_years.py` produces the figure demonstrating "pattern consistency over year-of-collection" for linear-position-distributions of function words. Follow the prompts to get the figure (as Fig. 2. of the paper).

`posdist_viz_meaning.py` produces the figure exhibiting "pattern consistency for words of similar meanings" for linear-position-distributions of function words. Follow the prompts to get the figure (as Fig. 3. of the paper).

## Extracting attention weights for words in sentences
Run `attn_rowpos.py` to get the attention weights and linear postions corresponding to words in sentences.

Usage:

`python attn_rowpos.py <sentence-length> <maximum-sentences-for-each-word> <corpus-path> <output-path> <words>`

 For one word in a sentence, BERT generate 12×12 attention rows. This script put generated attention rows and position labs (as a combined NumPy array) in separated directories (named after layer and head number):

```bash
~/output/00_00/data/attnrowlabs_00_00.pkl
~/output/00_01/data/attnrowlabs_00_01.pkl
...
~/output/12_12/data/attnrowlabs_12_12.pkl
```
## Probing linear-position-distribution from BERT
`pos_probe.py` depdends on the result of `attn_rowpos.py`, the outputs of the latter script serve as the input dataset to the former.

Usage:

`python pos_probe.py <train/test-split> <batch-size> <epochs> <learning-rate> <dataset-path>` 

Sample usage:

`python pos_probe.py 0.3 32 200 1e-3 ~/output/`




