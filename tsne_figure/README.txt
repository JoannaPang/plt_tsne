# preprocess the data, get 3 files: reid_4608_names.txt, reid_4608_labels.txt, reid_4608_features.txt
python preprocess3.py

# get the tsne embedding feature(2 dim) and save it to reid_4608_2d.mat, show the labeled fig
# cost a lot of time, some hours£¬ result 2dim result is attached
python tsne_reid.py

# show the embedding result by matlab
matlab -nodisplay
show_embedding_reid
