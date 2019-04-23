#处理负数版本
from sklearn import datasets
#digits = datasets.load_digits()
# Take the first 500 data points: it's hard to see 1500 points
import scipy.io as sio
import numpy as np
import copy
load_fn = 'lotsoffeatures/dqntemporalmarket1218ac11218temporaldense121duke_pytorch_result.mat'
#dqntemporalmarket1218ac11218temporaldense121vlids_pytorch_result 出现了负数，无法使用bincount函数
#dqntemporalmarket1218ac11218temporaldense121market_pytorch_result
#dqntemporalmarket1218ac11218temporaldense121duke_pytorch_result

#修改读入的mat文件名即可
load_data = sio.loadmat(load_fn) #以key-value的形式进行load
#print(load_data)
#print(type(load_data))
X = load_data['gallery_f'] #X为特征
y = load_data['gallery_label'] #y为label，即文件夹名，即pid
y = y.flatten() #对y进行降维，从二维降到一维
#print(type(X))
#print('x_dim')
#print(X.ndim)
#print('y_dim')
#print(y.ndim)
print(y)
y_2 = copy.deepcopy(y) #一定要采用深拷贝的方式，才能获得两个指针指向两个不同的地址，且其初始内容一样，且不会改变其中的一个而影响
print(len(y))
print(np.max(y_2))
maxofy_2 = np.max(y_2)+1
numnumnum=0
for i in range(len(y_2)):
    if y_2[i]<0:
        y_2[i]=maxofy_2
        print(y_2[i])
        numnumnum=numnumnum+1

print('yayayayay!')
print(numnumnum)

top15_list=[]
top15_list_value=[]
tempy=np.bincount(y_2)#把数组中出现的每个数字，当作index，数字出现的次数当作value来表示
#print(tempy)
for i in range(15):
    tempindex = np.argmax(tempy)  # 返回数组中最大值的下标
    top15_list.append(tempindex)
    top15_list_value.append(tempy[tempindex])
    #print(tempindex)
    tempy[tempindex] = -1

print('top15_id')
print(top15_list)
top15_list[0]=-1
print(top15_list)
print('top15_id_num')
print(top15_list_value)

#print(np.sum(y==3)) #统计y数组中3出现的次数
#print(tempy[473])



print('load_success!')
small_index = []
for i in range(len(top15_list)):
    #print(top15_list[i])
    kkkkk = np.where(y==top15_list[i])
    #print(type(kkkkk))#此时的kkkkk为(array([])),tuple里面包含了array数组
    kkkkk1 = np.array(kkkkk)#先将tuple数组转化为array
    kkkkk2 =kkkkk1.tolist()#将array转化为list
    #print(kkkkk2)
    #print(type(kkkkk2))
    #small_index.append(kkkkk2[0])#特别注意，该方法只是将其放在一起，并没有合并
    small_index=small_index+kkkkk2[0] #将list数组进行连接，+号最简单最容易
    #print(small_index)

print('final')
#print(small_index)

#print(type(small_index))

#qu10=[1,2,3,4,5,6,7,8,9,10]
#print(type(qu10))
X_small=X[small_index,:] #取numpy数组中的small_index包含行的元素
y_small=y[small_index]#取numpy数组中的small_index包含的元素
#print(X_small.shape)
#print(y_small.shape)
#print(y_small)
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
#labels = range(len(target_ids))
labels = range(1,16)
print('length')
print(len(labels))
from matplotlib import pyplot as plt
#plt.figure(figsize=(6, 5))
figsize = 6, 5
figure, ax = plt.subplots(figsize=figsize)

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
                    marker='^', edgecolors='black', linewidths=0.7, alpha=0.8)
    elif int(label)//10 == 2:
        plt.scatter(X_2d[y_small == i, 0], X_2d[y_small == i, 1], label=label,
                    marker='s', edgecolors='black', linewidths=0.7, alpha=0.8)
plt.legend()#图例和注记
plt.rcParams['font.family']='Times New Roman'  #所有的字体都是新罗马

plt.tick_params(labelsize=15)
'''
labelsXXX = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labelsXXX]
'''
plt.show()
