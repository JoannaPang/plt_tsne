
'''
from sklearn import datasets
digits = datasets.load_digits()
# Take the first 500 data points: it's hard to see 1500 points
X = digits.data[:500]
y = digits.target[:500]
print(type(X))
print(X.ndim)
print(X)
print(type(y))
print(y.ndim)
print(y)

'''
import scipy.io as sio
import numpy as np
#load_fn = 'PRW_result2.mat' #python实现load .mat文件，注意此时这里是一个dict类型的数据结构
load_fn = 'PRID_result2.mat'
load_data = sio.loadmat(load_fn) #以key-value的形式进行load
#print(load_data)
#print(type(load_data))
X = load_data['reidX'] #X为特征
#y = load_data['reidy'] #y为label，即文件夹名，即pid
target_names = load_data['target_names'] #图片的名称

print('XXX')
print(len(X))
X = np.delete(X, 0, 0) #删掉二维数组第0行
print(type(X))
print(X.ndim)
print(X)
'''
print('yyyyy')
print(len(y))
#y = y.reshape((1,6061))
y = y.flatten() #把二维数组降维为一维
print(type(y))
print(y.ndim)
print(y)
y = np.delete(y, 0, 0)
print(y)
print(len(y))
'''
z =np.array([659, 495, 45, 324, 284, 7, 482, 386, 343, 363])  #创建numpy类型的数组

print(z)
print(type(z))
print(len(z))
print(z.ndim) #保证z的维数为1


min_num = np.min(z) #获取numpy数组的最小值和最大值
max_num = np.max(z)
print(min_num,max_num)
#target_ids = range(len(y))
target_ids = range(min_num, max_num+1)
print(len(target_ids))
print(target_ids)

print('tttendend')

'''
print('namename')
print(type(target_names))
print(target_names)
target_names = np.delete(target_names, 0, 0)
print(target_names)
'''

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)

X_2d = tsne.fit_transform(X)

#target_ids = range(len(digits.target_names))

print('deal-end')

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
#for i, c, label in zip(target_ids, colors, y):
for i, c, label in zip(target_ids, colors, z):
#for i, c, label in zip(target_ids, colors, digits.target_names):
    #plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
    plt.scatter(X_2d[z == i, 0], X_2d[z == i, 1], label=z)
#plt.legend()
plt.show()


'''
list=['123','456']
temp_list_int = [int(x) for x in list]
print(temp_list_int)
'''