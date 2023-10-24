import matplotlib.pyplot as plt
import pickle
import argparse



def watch(word,len1,len2):
	wc1 = res[len1][word]
	density1 = [i/sum(wc1) for i in wc1]
	wc2 = res[len2][word]
	density2 = [i/sum(wc2) for i in wc2]
	num_pairs = [(i,j) for i in range(1,len1+1) for j in range(1,len2+1) if i<j]

	ok_pairs = []

	pos_pct = []

	cdensity1s = []
	cdensity2s = []

	for p in num_pairs:
		if p[0]/len1 == p[1]/len2:
			print(p)
			ok_pairs.append(p)
			
	for p in ok_pairs:
		print(f'{word}-{p[0]}/{len1}-{p[1]}/{len2}')
		print(sum(density1[:p[0]]))
		cdensity1s.append(sum(density1[:p[0]]))
		print(sum(density2[:p[1]]))
		cdensity2s.append(sum(density2[:p[1]]))
		pos_pct.append(p[0]/len1)

	fig = plt.figure()
	ax = fig.subplots()
	ax.plot(pos_pct,cdensity1s,marker='D')
	ax.plot(pos_pct,cdensity2s,marker='o')
	plt.show()
	plt.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('word')
	parser.add_argument('len1')
	parser.add_argument('len2')
	args = parser.parse_args()

	with open('./data/precalc/count/EN_43WORDS_LEN_4_TO_37.pkl', mode='rb') as file:
		res = pickle.load(file)

	watch(args.word, int(args.len1), int(args.len2))