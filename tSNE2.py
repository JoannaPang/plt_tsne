
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
load_fn = 'PRID_plt_TOP1-20_result2.mat'
load_data = sio.loadmat(load_fn) #以key-value的形式进行load
#print(load_data)
#print(type(load_data))
X = load_data['reidX'] #X为特征
y = load_data['reidy'] #y为label，即文件夹名，即pid
target_names = load_data['target_names'] #图片的名称

print('XXX')
print(len(X))
X = np.delete(X, 0, 0) #删掉二维数组第0行
print(type(X))
print(X.ndim)
print(X)

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
target_ids = range(len(z))
#target_ids = range(min_num, max_num+1)
print(len(target_ids))
print(target_ids)
'''


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
target_ids = np.array([61, 58, 67, 675, 11, 425, 372, 32, 406, 445, 494, 471, 392, 263, 383, 659, 495, 45, 324, 284,
                       7, 482, 386, 343, 363])

print('deal-end')

from matplotlib import pyplot as plt
import random
import operator

kkk = random.randint(0, 1)

#for kkk in range(0,25):


C = np.array([[255,0,0], [75,0,130], [30,144,255], [128,0,0], [178,34,34], [0,0,128], [255,200,0], [0,255,150], [255,0,255],
              [0,245,255], [255,255,0], [255,106,106], [200,0,128], [186,85,211], [0,128,255],[255,218,185], [245,222,179],
              [64,0,255], [230,230,250], [64, 255, 0], [255, 255, 255], [255,64,0], [147,112,219], [100,0,255],
              [255,200,200]])
#纯红，青null，道奇蓝null，栗色，耐火砖，海军蓝，null，薄荷奶油，洋红
#金null，黄null，淡红null，null，适中的兰花紫，桃色，小麦色
#null，薰衣草花的淡紫色，null，白色，null，紫色null，null
#null
#C = C.sort()
print(type(C))
print(C)

plt.figure(figsize=(6, 5))
labels = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23',
          '24','25']

for i, label, c in zip(target_ids, labels, C):
    if operator.ge(label,'300'):
        print(label)
        print(int(label))
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=label, c=c / 255.0, alpha=0.8, marker='^',
                    edgecolors='black',
                    linewidths=0.7)
    elif operator.lt(label,'300') and operator.ge(label, '200'):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=label, c=c / 255.0, alpha=0.8, marker='s',
                    edgecolors='black',
                    linewidths=0.7)
    else:
        print('lllll')
        print(label)
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=label, c=c/255.0, alpha=0.8, marker='o', edgecolors='black',
                linewidths=0.7)
    #plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=label, cmap=cm)

#plt.legend()
plt.show()


'''
list=['123','456']
temp_list_int = [int(x) for x in list]
print(temp_list_int)
'''
'''
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k','r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y'
#for i, c, label in zip(target_ids, colors, y):
for i, c, label in zip(target_ids, colors, labels):
#for i, c, label in zip(target_ids, colors, digits.target_names):
    #plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label, cmap=cm)

'''
#print(plt.cm.get_cmap('RdYlBu'))

#cm = plt.cm.get_cmap('RdYlBu')

#cm = plt.cm.get_cmap('Spectral',30)
#cm = colormap()
'''

import matplotlib as mpl
import matplotlib.colors as colors
# 自定义colormap
def colormap():
    # 白青绿黄红
    cdict = ['#FFFFFF', '#9ff113', '#5fbb44', '#f5f329', '#e50b32']
    # 按照上面定义的colordict，将数据分成对应的部分，indexed：代表顺序
    return colors.ListedColormap(cdict, 'indexed')

    #return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00','#FF0000', '#8B0000'], 256)
'''


'''
colors_pang = list(range(1, 20))

cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]


nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps)
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(cmap_category, cmap_list, nrows):
    fig, axes = plt.subplots(nrows=nrows)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

    for ax, name in zip(axes, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax.set_axis_off()


for cmap_category, cmap_list in cmaps:
    plot_color_gradients(cmap_category, cmap_list, nrows)

randlabel = np.random.randint(0,1,20)
z = randlabel
'''