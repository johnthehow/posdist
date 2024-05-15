import matplotlib.pyplot as plt
import pickle
import argparse
import numpy
import math



def pmf_cdf_plot(word,sentlen,data):
    plt.rcParams["font.sans-serif"]=["SimHei"]
    count_dist = data[sentlen][word]
    count_sum = sum(count_dist)
    xs = [i for i in range(1, len(count_dist)+1)]
    ys_pmf = [i/count_sum for i in count_dist]
    ys_cdf = [sum(count_dist[:i-1])/count_sum for i in xs]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    line_pmf, = ax1.plot(xs, ys_pmf, marker='D', lw=1, markerfacecolor='none', color='r', markersize=4, label='概率')
    line_cdf, = ax2.plot(xs, ys_cdf, marker='v',lw=1, markerfacecolor='none', color='b',markersize=4, label='累积概率')
    ax1.set_xticks(xs)
    ax1.set_xticklabels([i for i in range(1, len(count_dist)+1)])
    ax1.set_title(f'功能词: {word}')
    ax1.set_ylabel('概率', color='r')
    ax2.set_ylabel('累积概率', color='b')
    ax1.tick_params('y', colors='r')
    ax2.tick_params('y', colors='b')
    ax1.set_xlabel('句中线性位置')
    lines = [line_pmf, line_cdf]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center')
    plt.show()
    plt.close()
    return

def poslenq_cdf_plot(word,len1,len2,data):
    plt.rcParams["font.sans-serif"]=["SimHei"]
    wc1 = data[len1][word] # 词在句长a的位置频数分布
    density1 = [i/sum(wc1) for i in wc1] # 词在句长1的位置频率分布
    wc2 = data[len2][word] # 词在句长b的位置频数分布
    density2 = [i/sum(wc2) for i in wc2] # 词在句长2的位置频率分布
    num_pairs = [(i,j) for i in range(1,len1+1) for j in range(1,len2+1) if i<j] # 所有的可能的位置组合, 一个句长出一个位置

    ok_pairs = [] # 容器: 位置组合中第一个位置占句长a的比例 等于 位置组合中第二个位置占到句长b的比例, 则把位置组合加入
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
    ax.plot(pos_pct,cdensity1s,marker='D',label=len1)
    ax.plot(pos_pct,cdensity2s,marker='o',label=len2)
    ax.set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    ax.set_xticklabels([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    ax.set_yticklabels([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    ax.set_title(f'功能词: {word}')
    ax.set_xlabel('位置分位点q')
    ax.set_ylabel('累积概率')
    plt.legend(title='句长')
    plt.show()
    plt.close()
    return

def poslenq_cdf_baseline(word,len1,len2, data):
    plt.rcParams["font.sans-serif"]=["SimHei"]
    wc1 = data[len1][word] # 词在句长a的位置频数分布
    density1 = [i/sum(wc1) for i in wc1] # 词在句长1的位置频率分布
    wc2 = [int(sum(wc1)/len2)]*len2
    density2 = [i/sum(wc2) for i in wc2] # 词在句长2的位置频率分布
    num_pairs = [(i,j) for i in range(1,len1+1) for j in range(1,len2+1) if i<=j] # 所有的可能的位置组合, 一个句长出一个位置

    ok_pairs = [] # 容器: 位置组合中第一个位置占句长a的比例 等于 位置组合中第二个位置占到句长b的比例, 则把位置组合加入
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
    ax.plot(pos_pct,cdensity1s,marker='D',label=len1)
    ax.plot(pos_pct,cdensity2s,marker='o',label=len2)
    ax.set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    ax.set_xticklabels([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    ax.set_yticklabels([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    ax.set_title(f'功能词: {word}')
    ax.set_xlabel('位置分位点q')
    ax.set_ylabel('累积概率')
    plt.legend(title='句长')
    plt.show()
    plt.close()
    return

def poslenq_stat(word, sentlen_min, sentlen_max, data, output_filename):
    output_file = open(output_filename, mode='a', encoding='utf-8')
    output_file.write(f'word\tq1\tSL1\tq2\tSL2\tq\tcdf1\tcdf2\tcdf_diff\tmean_cdf_diff\tuniform_diff\tmean_uniform_diff\tplus_diff\tmean_plus_diff\tminus_diff\tmean_minus_diff\n')
    sentlen_pairs = quantile_points(sentlen_min, sentlen_max)
    print(len(sentlen_pairs.keys()))
    for sentlen_pair in sentlen_pairs.keys():
        print('-------')
        print(sentlen_pair)
        q1s = []
        q2s = []
        qs = []
        cdf1s = []
        cdf2s = []
        cdf2ps = []
        cdf2ms = []
        cdf_diffs = []
        uniform_diffs = []
        plus_diffs = []
        minus_diffs = []
        for q_pair in sentlen_pairs[sentlen_pair]:
            q1s.append(q_pair[0])
            q2s.append(q_pair[1])
            q = q_pair[0]/sentlen_pair[0]
            qs.append(q)
            cdf_1 = sum(data[sentlen_pair[0]][word][:q_pair[0]])/sum(data[sentlen_pair[0]][word])
            cdf1s.append(cdf_1)
            cdf_2 = sum(data[sentlen_pair[1]][word][:q_pair[1]])/sum(data[sentlen_pair[1]][word])
            cdf2s.append(cdf_2)
            cdf_2_plus = sum(data[sentlen_pair[1]][word][:q_pair[1]+1])/sum(data[sentlen_pair[1]][word])
            cdf2ps.append(cdf_2_plus)
            cdf_2_minus = sum(data[sentlen_pair[1]][word][:q_pair[1]-1])/sum(data[sentlen_pair[1]][word])
            cdf2ms.append(cdf_2_minus)
            print(f'{word}: ({q_pair[0]}/{sentlen_pair[0]}) ({q_pair[1]}/{sentlen_pair[1]}) q={q:.4f} cdf1={cdf_1:.4f} cdf2={cdf_2:.4f} diff={abs(cdf_2-cdf_1):.4f} diff_unif_0={abs(cdf_2-q_pair[0]/sentlen_pair[0]):.4f} diff_unif_1={abs(cdf_1-q_pair[0]/sentlen_pair[0]):.4f} diff_plus={abs(cdf_2_plus-cdf_1):.4f} diff_minus={abs(cdf_2_minus-cdf_1):.4f}')
            cdf_diffs.append(abs(cdf_2-cdf_1))
            uniform_diffs.append((abs(cdf_2-q)+abs(cdf_1-q))/2)
            plus_diffs.append(abs(cdf_2_plus-cdf_1))
            minus_diffs.append(abs(cdf_2_minus-cdf_1))
        mean_cdf_diff = sum(cdf_diffs)/len(cdf_diffs)
        mean_uniform_diff = sum(uniform_diffs)/len(uniform_diffs)
        mean_plus_diff = sum(plus_diffs)/len(plus_diffs)
        mean_minus_diff = sum(minus_diffs)/len(minus_diffs)
        
        # print(f'{sentlen_pair}: cdf_diff={mean_cdf_diff:.4f} uniform_0_diff={mean_uniform_0_diff:.4f} uniform_1_diff={mean_uniform_1_diff:.4f} plus_diff={mean_plus_diff:.4f} minus_diff={mean_minus_diff:.4f}')
        for (q1,q2,q,cdf1,cdf2,cdf_diff, unif_diff, plus_diff, minus_diff) in zip(q1s,q2s,qs,cdf1s,cdf2s,cdf_diffs,uniform_diffs,plus_diffs,minus_diffs):
            output_file.write(f'{word}\t{q1}\t{sentlen_pair[0]}\t{q2}\t{sentlen_pair[1]}\t{q:.4f}\t{cdf1:.4f}\t{cdf2:.4f}\t{cdf_diff:.4f}\t{mean_cdf_diff:.4f}\t{unif_diff:.4f}\t{mean_uniform_diff:.4f}\t{plus_diff:.4f}\t{mean_plus_diff:.4f}\t{minus_diff:.4f}\t{mean_minus_diff:.4f}\n')
    output_file.close()
    return

def quantile_points(sentlen_min, sentlen_max):
    sentlen_pairs = [(i,j) for i in range(sentlen_min,sentlen_max+1) for j in range(sentlen_min,sentlen_max+1) if i<j]
    result = {}
    for p in sentlen_pairs:
        len1, len2 = p
        equa_pairs = [(i,j) for i in range(1,len1+1) for j in range(1,len2+1) if i<j and i/len1 == j/len2]
        if len(equa_pairs) >= 4:
            result[(len1,len2)]=equa_pairs[:-1]
    return result

# 用例 shade('and',16,24,24, 36, data)
def shade(word, pos1, len1, pos2, len2,  data): # 绘制两个子图, 阴影图, 两个句长到相同比例位置, 累积分布概率相同
    plt.rcParams["font.sans-serif"]=["SimHei"]
    assert pos1/len1 == pos2/len2, "not a equiratio-position" # 位置1占句长1的比例 等于 位置2占句长2的比例
    fig = plt.figure()
    ax = fig.subplots()
    ax0_xs = [i for i in range(1,len1+1)]
    ax0_ys = data[len1][word]
    ax0_ys = [i/sum(ax0_ys) for i in ax0_ys]
    cum_prob_0 = sum(ax0_ys[:pos1])

    ax1_xs = [i for i in range(1,len2+1)]
    ax1_ys = data[len2][word]
    ax1_ys = [i/sum(ax1_ys) for i in ax1_ys]
    cum_prob_1 = sum(ax1_ys[:pos2])    

    ax.set_xticks(ax1_xs)
    ax.set_xticklabels(ax1_xs,rotation=90)
    ax.plot(ax0_xs,ax0_ys, label=len1)
    ax.plot(ax1_xs,ax1_ys, label=len2)
    ax.fill_between(ax0_xs[:pos1],ax0_ys[:pos1],color='skyblue', alpha=0.4, label=f'句长{len1}累积概率: {cum_prob_0:.3f}')
    ax.fill_between(ax1_xs[:pos2],ax1_ys[:pos2],color='pink', alpha=0.4, label=f'句长{len2}累积概率: {cum_prob_1:.3f}')
    ax.set_title(f'功能词: {word}')
    ax.set_ylabel('概率')
    ax.set_xlabel('句中线性位置')
    plt.legend(title='句长')
    plt.show()
    plt.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('word')
    parser.add_argument('len1')
    parser.add_argument('len2')
    # parser.add_argument('pos1')
    # parser.add_argument('pos2')
    parser.add_argument('data_path')
    args = parser.parse_args()

    args.word = 'but'
    args.pos1 = '16'
    args.len1 = '30'
    args.pos2 = '24'
    args.len2 = '36'
    args.data_path = r'C:\Program Files\Python37\Lib\site-packages\thehow\posdist\data\precalc\count\EN_43WORDS_LEN_4_TO_37.pkl'
    with open(args.data_path, mode='rb') as file:
        data = pickle.load(file) # 词在各个位置的频数分布

    # # 用于产生在各个 位置/句长 等比点 的两句长累积分布曲线
    # poslenq_cdf_plot(args.word, int(args.len1), int(args.len2), data)

    # # 用于产生示意图: 两个句长的线性位置分布, 在等比位置下的累积分布(表现为阴影面积)
    # shade(args.word, int(args.pos1), int(args.len1), int(args.pos2), int(args.len2), data)

    # res = quantile_points(4,37)

    # cdf(args.word, int(args.len1), data)

    # pmf_cdf_plot(args.word, int(args.len1), data)
    # wordlist = ['the', 'a', 'an', 'to', 'of', 'in', 'for', 'with', 'on', 'as', 'at', 'from', 'by', 'about', 'into', 'than', 'over', 'after', 'before', 'i', 'you', 'it', 'we', 'they', 'he', 'them', 'him', 'she', 'me', 'us', 'that', 'this', 'who', 'which', 'what', 'and', 'or', 'but', 'if', 'so', 'because', 'while']
    # for word in wordlist:
    #     poslenq_stat(word, int(args.len1), int(args.len2), data, r'D:/test/output.txt')


    # poslenq_cdf_baseline('this', 24, 36, data)
    poslenq_cdf_plot('this', 24, 36, data)
    print('done')