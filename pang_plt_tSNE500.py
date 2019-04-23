from sklearn import datasets
#digits = datasets.load_digits()
# Take the first 500 data points: it's hard to see 1500 points
import scipy.io as sio
import numpy as np
load_fn = 'lotsoffeatures/dqntemporalprw1218ac11218temporaldense121vlids_pytorch_result.mat'
#修改读入的mat文件名即可
load_data = sio.loadmat(load_fn) #以key-value的形式进行load
#print(load_data)
#print(type(load_data))
X = load_data['gallery_f'] #X为特征
y = load_data['gallery_label'] #y为label，即文件夹名，即pid
y = y.flatten() #对y进行降维，从二维降到一维
#print('x_dim')
#print(X.ndim)
#print('y_dim')
#print(y.ndim)
#print(y)

print('load_success!')

#qu10=[1,2,3,4,5,6,7,8,9,10]
X_small=X[0:500,:] #取numpy数组中的前500行元素
y_small=y[0:500]#取numpy数组中的前500个元素
#print(X_small.shape)
#print(y_small.shape)
print(y_small)
print('samll_complete!')

#X = digits.data[:500]
#y = digits.target[:500]

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)

X_2d = tsne.fit_transform(X_small)
#print('whywhathow!')
#print(X_2d.ndim)
#print(X_2d)
#target_ids = range(len(digits.target_names))

#target_ids = range(len(y))
#target_ids = np.array([5985,838])
target_ids={}.fromkeys(y_small).keys() #过滤掉数组中重复的元素（重要！！！）
print(target_ids)
labels = range(len(target_ids))
print('length')
print(len(labels))
from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, label in zip(target_ids, labels):
    #print('time1')
    #print(i)
    #print(label)
    if int(label)//10 == 0:
        plt.scatter(X_2d[y_small == i, 0], X_2d[y_small == i, 1], label=label,
                    marker='o', edgecolors='black', linewidths=0.7, alpha=0.8)
    elif int(label)//10 == 1:
        plt.scatter(X_2d[y_small == i, 0], X_2d[y_small == i, 1], label=label,
                    marker='s', edgecolors='black', linewidths=0.7, alpha=0.8)
    elif int(label)//10 == 2:
        plt.scatter(X_2d[y_small == i, 0], X_2d[y_small == i, 1], label=label,
                    marker='^', edgecolors='black', linewidths=0.7, alpha=0.8)
#plt.legend()#图例和注记
plt.show()
