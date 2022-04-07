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
	fig = plt.figure(figsize = (8,6),dpi=300)
	axe = fig.subplots()
	for i in range(1,13):
		axe.scatter(acc[acc.layer==i].layer,acc[acc.layer==i].acc,s=12)
		layer_means.append(acc[acc.layer==i].mean().acc)
	axe.plot(range(1,13),layer_means,marker='^',label='layer mean accuracy')
	axe.set_xticks(range(1,13))
	axe.set_xlabel('attention layer')
	axe.set_ylabel('accuracy')
	axe.set_ylim(0,1.05)
	axe.plot([1,12],[baseline,baseline],color='k',ls='--',linewidth=1)
	axe.text(3.5,baseline-0.034,f'local majority baseline: {baseline}')
	plt.legend()
	now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	plt.savefig(acc_path.joinpath(f'probe_acc_{now}.png'),format='png')
	print(f'Plot saved as {acc_path.joinpath(now)}.png')