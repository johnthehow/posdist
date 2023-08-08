import pickle
import matplotlib.pyplot as plt

with open('C:/Users/dell/Desktop/thehow/TASKS/POSDIST/ATTNROWLABS/12/03_01/data/attnrowlabs_03_01.pkl',mode='rb') as file:
	and_attn_rowlabs_h0301_12 = pickle.load(file)
with open('C:/Users/dell/Desktop/thehow/TASKS/POSDIST/ATTNROWLABS/16/03_01/data/attnrowlabs_03_01.pkl',mode='rb') as file:
	and_attn_rowlabs_h0301_16 = pickle.load(file)
with open('C:/Users/dell/Desktop/thehow/TASKS/POSDIST/ATTNROWLABS/20/03_01/data/attnrowlabs_03_01.pkl',mode='rb') as file:
	and_attn_rowlabs_h0301_20 = pickle.load(file)
with open('C:/Users/dell/Desktop/thehow/TASKS/POSDIST/ATTNROWLABS/24/03_01/data/attnrowlabs_03_01.pkl',mode='rb') as file:
	and_attn_rowlabs_h0301_24 = pickle.load(file)
with open('C:/Users/dell/Desktop/thehow/TASKS/POSDIST/ATTNROWLABS/28/03_01/data/attnrowlabs_03_01.pkl',mode='rb') as file:
	and_attn_rowlabs_h0301_28 = pickle.load(file)
with open('C:/Users/dell/Desktop/thehow/TASKS/POSDIST/ATTNROWLABS/32/03_01/data/attnrowlabs_03_01.pkl',mode='rb') as file:
	and_attn_rowlabs_h0301_32 = pickle.load(file)
with open('C:/Users/dell/Desktop/thehow/TASKS/POSDIST/ATTNROWLABS/36/03_01/data/attnrowlabs_03_01.pkl',mode='rb') as file:
	and_attn_rowlabs_h0301_36 = pickle.load(file)

with open('C:/Users/dell/Desktop/thehow/TASKS/POSDIST/ATTNROWLABS/12/03_10/data/attnrowlabs_03_10.pkl',mode='rb') as file:
	and_attn_rowlabs_h0310_12 = pickle.load(file)
with open('C:/Users/dell/Desktop/thehow/TASKS/POSDIST/ATTNROWLABS/16/03_10/data/attnrowlabs_03_10.pkl',mode='rb') as file:
	and_attn_rowlabs_h0310_16 = pickle.load(file)
with open('C:/Users/dell/Desktop/thehow/TASKS/POSDIST/ATTNROWLABS/20/03_10/data/attnrowlabs_03_10.pkl',mode='rb') as file:
	and_attn_rowlabs_h0310_20 = pickle.load(file)
with open('C:/Users/dell/Desktop/thehow/TASKS/POSDIST/ATTNROWLABS/24/03_10/data/attnrowlabs_03_10.pkl',mode='rb') as file:
	and_attn_rowlabs_h0310_24 = pickle.load(file)
with open('C:/Users/dell/Desktop/thehow/TASKS/POSDIST/ATTNROWLABS/28/03_10/data/attnrowlabs_03_10.pkl',mode='rb') as file:
	and_attn_rowlabs_h0310_28 = pickle.load(file)
with open('C:/Users/dell/Desktop/thehow/TASKS/POSDIST/ATTNROWLABS/32/03_10/data/attnrowlabs_03_10.pkl',mode='rb') as file:
	and_attn_rowlabs_h0310_32 = pickle.load(file)
with open('C:/Users/dell/Desktop/thehow/TASKS/POSDIST/ATTNROWLABS/36/03_10/data/attnrowlabs_03_10.pkl',mode='rb') as file:
	and_attn_rowlabs_h0310_36 = pickle.load(file)

and_attn_row_h0301_12 = and_attn_rowlabs_h0301_12[:,:12]
and_attn_row_h0301_16 = and_attn_rowlabs_h0301_16[:,:16]
and_attn_row_h0301_20 = and_attn_rowlabs_h0301_20[:,:20]
and_attn_row_h0301_24 = and_attn_rowlabs_h0301_24[:,:24]
and_attn_row_h0301_28 = and_attn_rowlabs_h0301_28[:,:28]
and_attn_row_h0301_32 = and_attn_rowlabs_h0301_32[:,:32]
and_attn_row_h0301_36 = and_attn_rowlabs_h0301_36[:,:36]

and_attn_row_h0310_12 = and_attn_rowlabs_h0310_12[:,:12]
and_attn_row_h0310_16 = and_attn_rowlabs_h0310_16[:,:16]
and_attn_row_h0310_20 = and_attn_rowlabs_h0310_20[:,:20]
and_attn_row_h0310_24 = and_attn_rowlabs_h0310_24[:,:24]
and_attn_row_h0310_28 = and_attn_rowlabs_h0310_28[:,:28]
and_attn_row_h0310_32 = and_attn_rowlabs_h0310_32[:,:32]
and_attn_row_h0310_36 = and_attn_rowlabs_h0310_36[:,:36]

and_attn_row_h0301_12_mean = and_attn_row_h0301_12.mean(axis=0).tolist()
and_attn_row_h0301_16_mean = and_attn_row_h0301_16.mean(axis=0).tolist()
and_attn_row_h0301_20_mean = and_attn_row_h0301_20.mean(axis=0).tolist()
and_attn_row_h0301_24_mean = and_attn_row_h0301_24.mean(axis=0).tolist()
and_attn_row_h0301_28_mean = and_attn_row_h0301_28.mean(axis=0).tolist()
and_attn_row_h0301_32_mean = and_attn_row_h0301_32.mean(axis=0).tolist()
and_attn_row_h0301_36_mean = and_attn_row_h0301_36.mean(axis=0).tolist()

and_attn_row_h0310_12_mean = and_attn_row_h0310_12.mean(axis=0).tolist()
and_attn_row_h0310_16_mean = and_attn_row_h0310_16.mean(axis=0).tolist()
and_attn_row_h0310_20_mean = and_attn_row_h0310_20.mean(axis=0).tolist()
and_attn_row_h0310_24_mean = and_attn_row_h0310_24.mean(axis=0).tolist()
and_attn_row_h0310_28_mean = and_attn_row_h0310_28.mean(axis=0).tolist()
and_attn_row_h0310_32_mean = and_attn_row_h0310_32.mean(axis=0).tolist()
and_attn_row_h0310_36_mean = and_attn_row_h0310_36.mean(axis=0).tolist()

fig = plt.figure(dpi=300)
ax = fig.subplots()

ax.plot([i+1 for i in range(12)], and_attn_row_h0301_12_mean, label='12', marker='+')
ax.plot([i+1 for i in range(16)], and_attn_row_h0301_16_mean, label='16', marker='x')
ax.plot([i+1 for i in range(20)], and_attn_row_h0301_20_mean, label='20', marker='D')
ax.plot([i+1 for i in range(24)], and_attn_row_h0301_24_mean, label='24', marker='s')
ax.plot([i+1 for i in range(28)], and_attn_row_h0301_28_mean, label='28', marker='o')
ax.plot([i+1 for i in range(32)], and_attn_row_h0301_32_mean, label='32', marker='^')
ax.plot([i+1 for i in range(36)], and_attn_row_h0301_36_mean, label='36', marker='v')
ax.set_xticks([i for i in range(1,37)])
ax.set_xticklabels([i for i in range(1,37)],rotation='vertical')
ax.set_xlabel('linear position in sentence')
ax.set_ylabel('attention score')
ax.legend(title = 'Sent Len')

plt.savefig(r'C:\Users\dell\Desktop\thehow\TASKS\POSDIST\VIS\0301\and_h0301_12_36.png', format='png')
plt.clf()

fig = plt.figure(dpi=300)
ax = fig.subplots()

ax.plot([i+1 for i in range(12)], and_attn_row_h0301_12_mean, label='12', marker='+')
ax.plot([i+1 for i in range(16)], and_attn_row_h0301_16_mean, label='16', marker='x')
ax.plot([i+1 for i in range(20)], and_attn_row_h0301_20_mean, label='20', marker='D')
ax.plot([i+1 for i in range(24)], and_attn_row_h0301_24_mean, label='24', marker='s')
ax.plot([i+1 for i in range(28)], and_attn_row_h0301_28_mean, label='28', marker='o')
ax.plot([i+1 for i in range(32)], and_attn_row_h0301_32_mean, label='32', marker='^')
ax.plot([i+1 for i in range(36)], and_attn_row_h0301_36_mean, label='36', marker='v')
ax.set_xticks([i for i in range(1,37)])
ax.set_xticklabels([i for i in range(1,37)],rotation='vertical')
ax.set_xlabel('linear position in sentence')
ax.set_ylabel('attention score')
ax.legend(title = 'Sent Len')

plt.savefig(r'C:\Users\dell\Desktop\thehow\TASKS\POSDIST\VIS\0310\and_h0310_12_36.png', format='png')
plt.clf()