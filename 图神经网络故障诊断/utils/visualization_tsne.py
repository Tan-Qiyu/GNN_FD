import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.manifold import TSNE
from utils.tsne import plot_embedding

def visualization_tsne(loader_test,y_feature,classes):

    #the label of testing dataset
    label = np.empty(0,)
    for i in range(len(loader_test.dataset)):
        label = np.append(label,loader_test.dataset[i].y)

    #tsne
    warnings.filterwarnings('ignore')
    tsne = TSNE(n_components=2, init='pca')

    result = tsne.fit_transform(y_feature)  # 对特征进行降维
    fig = plot_embedding(result, label,classes=classes)

