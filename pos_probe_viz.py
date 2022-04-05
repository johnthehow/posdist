import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import datetime



if __name__ == '__main__':
	acc_path = Path(sys.argv[1])
	acc = pd.read_csv(acc_path.joinpath('accuracies.txt'),sep='\t')
	filelist = os.listdir(acc_path)
	for fname in filelist:
		if fname.find('baseline_') != -1:
			baseline = float(fname[10:])

	layer_means = []
	for i in range(1,13):
		plt.scatter(acc[acc.layer==i].layer,acc[acc.layer==i].acc,s=12)
		layer_means.append(acc[acc.lay==i].mean().acc)
	plt.plot(range(1,13),layer_means,marker='^',label='layer mean accuracy')
	plt.plot([1,12],[baseline,baseline],color='k',ls='--',linewidth=1)
	plt.text(4,baseline-0.034,f'local majority baseline: {baseline}')
	plt.legend()
	now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	plt.savefig(save_path.joinpath(f'probe_acc_{now}.png'),format='png')
	print(f'Plot saved as {save_path.joinpath(now)}.png')