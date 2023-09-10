import pickle
import matplotlib.pyplot as plt

words = ['and', 'für', 'les', 'pero', 'на', 'nebo']
langs = ['English','German','French','Spanish','Russian','Czech']
lens = [12, 16, 20, 24, 28, 32, 36]

density_database_paths = [
"../data/precalc/density/en_43words_4_37.pkl",
"../data/precalc/density/de_38words_4_37.pkl",
"../data/precalc/density/fr_41words_4_37.pkl",
"../data/precalc/density/es_39words_4_37.pkl",
"../data/precalc/density/ru_39words_4_37.pkl",
"../data/precalc/density/cz_37words_4_37.pkl"
]

density_databases = []

for p in density_database_paths:
	with open(p, mode='rb') as file:
		density_databases.append(pickle.load(file))

# plt.rcParams["font.sans-serif"]=["SimHei"] 
fig = plt.figure(figsize=(15,16),dpi=300, tight_layout=True)

axes = fig.subplots(3,2).flatten()
props = dict(boxstyle='square', facecolor='white', alpha=0.5)

for img in range(6):
	linestyle_cnt = iter(range(7))
	marker = iter(['+','x','D','s','o','^','v'])
	for length in range(12,37,4):
		ys = density_databases[img][length][words[img]]
		xs = [i+1 for i in range(len(ys))]
		axes[img].plot(xs,ys, marker = next(marker), label=f'{length}', fillstyle='none', linewidth=1, linestyle=(0,(7,next(linestyle_cnt))))
		axes[img].set_xticks(range(1,37))
		axes[img].set_xticklabels(range(1,37), rotation='vertical')
		axes[img].tick_params(labelsize=16)
		axes[img].set_xlabel('linear position in sentence', fontsize=18)
		axes[img].set_ylabel('probability', fontsize=18)
		axes[img].text(0.5, 0.95, f'{words[img]}\n({langs[img]})', horizontalalignment='center',transform=axes[img].transAxes, fontsize=14, verticalalignment='top', bbox=props)
	axes[img].legend(title='Sent Len', fontsize=17, title_fontsize=17)
plt.savefig('../images/length_stable.png',format='png')
plt.close()