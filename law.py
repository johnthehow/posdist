import matplotlib.pyplot as plt
import pickle
import argparse


def match(word,len1,len2):
	wc1 = res[len1][word] # 词在句长1的位置频数分布
	density1 = [i/sum(wc1) for i in wc1] # 词在句长1的位置频率分布
	wc2 = res[len2][word] # 词在句长2的位置频数分布
	density2 = [i/sum(wc2) for i in wc2] # 词在句长2的位置频率分布
	num_pairs = [(i,j) for i in range(1,len1+1) for j in range(1,len2+1) if i<j] # 所有的可能的位置组合, 一个句长出一个位置

	ok_pairs = [] # 容器: 位置组合中第一个位置占句长1的比例 等于 位置组合中第二个位置占到句长2的比例, 则把位置组合加入
	pos_pct = [] # 容器: 位置分为点, 两个句长中是一样的

	cdensity1s = []
	cdensity2s = []

	for p in num_pairs:
		if p[0]/len1 == p[1]/len2: # 如果位置组合中第一个位置占句长1的比例 等于 位置组合中第二个位置占到句长2的比例
			print(p)
			ok_pairs.append(p)
			
	for p in ok_pairs:
		print(f'{word}-{p[0]}/{len1}-{p[1]}/{len2}')
		print(sum(density1[:p[0]]))
		cdensity1s.append(sum(density1[:p[0]])) # 在句长1中, 到位置1的累积概率
		print(sum(density2[:p[1]]))
		cdensity2s.append(sum(density2[:p[1]])) # 在句长2中, 到为止2的累积概率
		pos_pct.append(p[0]/len1)

	fig = plt.figure()
	ax = fig.subplots()
	ax.plot(pos_pct,cdensity1s,marker='D',label=f'sentlen{len1}')
	ax.plot(pos_pct,cdensity2s,marker='o',label=f'sentlen{len2}')
	ax.set_title(f'word: {word}')
	ax.set_xlabel('position quantile')
	ax.set_ylabel('cumulative probability')
	plt.legend()
	plt.show()
	plt.close()

# 用例 shade_overlap('and',24,36,16,24)
def shade_overlap(word, len1, len2, pos1, pos2): # 绘制两个子图, 阴影图, 两个句长到相同比例位置, 累积分布概率相同
	fig = plt.figure()
	ax = fig.subplots()
	ax0_xs = [i for i in range(1,len1+1)]
	ax0_ys = res[len1][word]
	ax0_ys = [i/sum(ax0_ys) for i in ax0_ys]
	cum_prob_0 = sum(ax0_ys[:pos1])

	ax1_xs = [i for i in range(1,len2+1)]
	ax1_ys = res[len2][word]
	ax1_ys = [i/sum(ax1_ys) for i in ax1_ys]
	cum_prob_1 = sum(ax1_ys[:pos2])	

	ax.set_xticks(ax1_xs)
	ax.set_xticklabels(ax1_xs,rotation=90)
	ax.plot(ax0_xs,ax0_ys)
	ax.plot(ax1_xs,ax1_ys)
	ax.fill_between(ax0_xs[:pos1],ax0_ys[:pos1],color='skyblue', alpha=0.4, label='Shaded Area')
	ax.fill_between(ax1_xs[:pos2],ax1_ys[:pos2],color='pink', alpha=0.4, label='Shaded Area')
	ax.set_title(f'word: {word}, sentlen: {len1},{len2}, cum_prob1: {cum_prob_0:.3f}, cum_prob2:{cum_prob_1:.3f}')
	plt.show()
	plt.close()
	return

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('word')
	parser.add_argument('len1')
	parser.add_argument('len2')
	args = parser.parse_args()

	with open('./data/precalc/count/EN_43WORDS_LEN_4_TO_37.pkl', mode='rb') as file:
		res = pickle.load(file) # 词在各个位置的频数分布

	# 用于产生在各个 位置/句长 等比点 的两句长累积分布曲线
	match(args.word, int(args.len1), int(args.len2))

	# 用于产生示意图: 两个句长的线性位置分布, 在等比位置下的累积分布(表现为阴影面积)
	# shade_overlap(args.word, int(args.len1), int(args.len2), 16, 24)