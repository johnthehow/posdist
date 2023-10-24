import pickle
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False

def draw():
	with open('../data/specific/crossling_stable.pkl', mode='rb') as file:
		database = pickle.load(file)
	langs = ['en', 'de', 'fr', 'es', 'ru', 'cz']
	langs_full = ['英语', '德语', '法语', '西班牙语', '俄语', '捷克语']
	groups = ['grp1', 'grp2', 'grp3', 'grp4']
	wordlists = [['and', 'und', 'et', 'y', 'и', 'a'], ['or', 'oder', 'ou', 'o', 'или', 'nebo'], ['i', 'ich', 'je', 'yo', 'я', 'já'], ['we', 'wir', 'nous', 'nosotros', 'мы', 'my']]

	fig = plt.figure(figsize = (16,14),dpi=300)
	axes = fig.subplots(2,2).flatten()


	grp_cnt = 0
	for grp in groups:
		markers = iter(['+', 'x', 'D', 's', 'o', '^', 'v'])
		linestyle_cnt = iter(range(7))
		lang_cnt = 0
		for lang in langs:
			ys = database[grp][lang]
			xs = [i+1 for i in range(len(ys))]
			xticks = [i for i in range(1,len(xs)+1)]
			axes[grp_cnt].plot(xs,ys, label = wordlists[grp_cnt][lang_cnt] + f' ({langs_full[lang_cnt]})', marker=next(markers), fillstyle='none', linewidth=1, linestyle=(0,(7,next(linestyle_cnt))))
			lang_cnt += 1
		axes[grp_cnt].set_xticks(xticks)
		axes[grp_cnt].tick_params(labelsize=18)
		axes[grp_cnt].set_ylabel('概率', fontsize=18)
		axes[grp_cnt].set_xlabel('目标词句中线性位置', fontsize=18)
		axes[grp_cnt].legend(title='功能词', fontsize=18, title_fontsize=18)
		grp_cnt += 1
	axes[0].legend(title='功能词', fontsize=18, title_fontsize=18, loc=(0.4,0.03))
	plt.savefig('../images/crossling_stable_cn.png',format='png')
	plt.close()

draw()