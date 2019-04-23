import os
import re
import cv2
import mat4py
import numpy as np
import heapq
import scipy.io as sio

def F_walk_files(base_dir, pattern, dirs = False):
    paths = []
    for root, dirs, files in os.walk(base_dir, followlinks=True):
        if dirs:
            items = dirs
        else:
            items = files
        for item in items:
            if re.match(pattern, item) is not None:
                paths.append(os.path.join(root, item))
    return paths


# jpg dir
target_files = F_walk_files("/home/liuliang/DISK_2T/others/figure/clusterresult", r".*jpg")
# small jpg size
H = W = 30
# small jpg saved dir
target_dir = "/home/liuliang/DISK_2T/others/figure/small_jpg"
# feature mat
X = mat4py.loadmat("/home/liuliang/DISK_2T/others/figure/swf_feature2.mat")

all_names = []
all_features = []
all_labels = []


all_names.extend(X['query_names'])
all_names.extend(X['gallery_names'])
all_names.extend(X['train_all_names'])

all_labels.extend(X['query_label'])
all_labels.extend(X['gallery_label'])
all_labels.extend(X['train_all_label'])

all_features.extend(X['query_f'])
all_features.extend(X['gallery_f'])
all_features.extend(X['train_all_f'])

all_names = [name.split('/')[-1] for name in all_names]
all_labels = np.array(all_labels)
all_features = np.array(all_features)

# get the features of existed file
error = 0
exists_paths = []
exists_labels = []
exists_f = []
for img_path in target_files:
    img_name = img_path.split('/')[-1]
    if img_name not in all_names:
        error +=1 
        print("Error:{}".format(error))
    else:
        exists_paths.append(img_path)
        ind = all_names.index(img_name)
        exists_labels.append(all_labels[ind])
        exists_f.append(all_features[ind])

exists_labels = np.array(exists_labels)
exists_f = np.array(exists_f)

# top n labels
tmp_pairs = []
for lab in range(1, 934):
    ind = np.where(exists_labels == lab)
    tmp_pairs.append((ind, lab))

top_n = heapq.nlargest(100, tmp_pairs, key = lambda x:len(x[0][0]))

valid_names = []
valid_labels = []
valid_f = []

#print(top_n)
# samples with top n labels
i = 0
for item in top_n:
    cur_lab = item[1]
    print("label={} cnt={}".format(item[1], len(item[0][0])))
    for ind in item[0][0]: 
        i += 1
        name = exists_paths[ind]
        lab = exists_labels[ind]
        print("[{}] {}".format(i, name))
        valid_names.append(name)
        valid_f.append(exists_f[ind])
        valid_labels.append(lab)
        assert(lab == cur_lab)
        assert(name in target_files)

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

with open('reid_4608_names.txt','w') as f:
    for name in valid_names:
        img = cv2.imread(name)

        h, w, c = img.shape
        ratio = min(float(W) / w, float(H) / h)

        small_img = cv2.resize(img, (int(ratio * w), int(ratio * h)), cv2.INTER_CUBIC)
        target_path = os.path.join(target_dir, name.split('/')[-1])
        cv2.imwrite(target_path, small_img)
        f.write('{}\n'.format(target_path))

np.savetxt("reid_4608_labels.txt", np.array(valid_labels))
np.savetxt("reid_4608_features.txt", np.array(valid_f))

#sio.savemat("reid_4608_labels.mat", {'labels': np.array(valid_labels)})
#sio.savemat("reid_4608_features.mat", {'X': np.array(valid_f)})

