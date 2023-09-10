import matplotlib.pyplot as plt
import pickle

with open('../data/specific/year_stable.pkl', mode='rb') as file:
	image_data = pickle.load(file)

words = ['but', 'auf', 'los']
langs = ['English', 'German', 'Spanish']
years_en = [2005, 2009, 2013, 2017]
years_de = [2007, 2011, 2015, 2019]
years_es = [2008, 2012, 2016, 2020]
years_all = [years_en, years_de, years_es]

fig = plt.figure(figsize=(17.5,5.2),dpi=300, tight_layout=False)
axes = fig.subplots(1,3).flatten()
props = dict(boxstyle='square', facecolor='white', alpha=0.5)
word_cnt = 0
for word in words: # 每种语言的一个单词
	markers = iter(['+', 'x', 'D', 's', 'o', '^', 'v'])
	linestyle_cnt = iter(range(7))
	year_cnt = 0
	for year in range(4):
		axes[word_cnt].plot(image_data[word_cnt][year_cnt][0],image_data[word_cnt][year_cnt][1], label=years_all[word_cnt][year_cnt], marker=next(markers), fillstyle='none', linewidth=1, linestyle=(0,(7,next(linestyle_cnt))))
		axes[word_cnt].set_xticks(image_data[word_cnt][year_cnt][0])	
		axes[word_cnt].set_xticklabels(image_data[word_cnt][year_cnt][0],rotation='vertical')
		axes[word_cnt].tick_params(labelsize=14)
		axes[word_cnt].set_xlabel('linear position in sentence', fontsize=14)
		axes[word_cnt].set_ylabel('probability', fontsize=14)
		year_cnt += 1
	axes[word_cnt].legend(title='year', fontsize=14, title_fontsize=14)
	axes[word_cnt].text(0.5, 0.95, f'{words[word_cnt]}\n({langs[word_cnt]})', horizontalalignment='center',transform=axes[word_cnt].transAxes, fontsize=14, verticalalignment='top', bbox=props)
	word_cnt += 1
axes[2].legend(title='year', fontsize=14, title_fontsize=14, loc=(0.35,0.02))
fig.subplots_adjust(wspace=0.2)
plt.savefig('../images/year_stable.png',format='png')
plt.close()

