import pickle
import matplotlib.pyplot as plt

def draw():
	with open('crossling_stable_20230911010449.pkl', mode='rb') as file:
		database = pickle.load(file)
	langs = ['en', 'de', 'fr', 'es', 'ru', 'cz']
	langs_full = ['English', 'German', 'French', 'Spanish', 'Russian', 'Czech']
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
		axes[grp_cnt].set_ylabel('probability', fontsize=18)
		axes[grp_cnt].set_xlabel('linear position in sentence', fontsize=18)
		axes[grp_cnt].legend(title='word', fontsize=18, title_fontsize=18)
		grp_cnt += 1
	axes[0].legend(title='word', fontsize=18, title_fontsize=18, loc=(0.4,0.03))
	plt.savefig('crossling_stable.png',format='png')
	plt.close()

draw()