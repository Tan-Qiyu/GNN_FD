import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.confusion import confusion

def visualization_confusion(loader_test,prediction):

    #the label of testing dataset
    label = np.empty(0,)
    for i in range(len(loader_test.dataset)):
        label = np.append(label,loader_test.dataset[i].y)

    #confusion matrix
    confusion_data = confusion_matrix(label,prediction)

    confusion(confusion_matrix=confusion_data)  #混淆矩阵绘制



