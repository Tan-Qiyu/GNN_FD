import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager

# 设置全局字体及大小，设置公式字体
config = {
    "font.family":'serif',        # 衬线字体
    "font.size": 12,              # 相当于小四大小
    "mathtext.fontset":'stix',    # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    "font.serif": ['SimSun'],     # 宋体SimSun
    "axes.unicode_minus": False,  # 用来正常显示负号
    "xtick.direction":'in',       # 横坐标轴的刻度设置向内(in)或向外(out)
    "ytick.direction":'in',       # 纵坐标轴的刻度设置向内(in)或向外(out)
}
rcParams.update(config)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimSun']  #显示中文

# %matplotlib inline
# 内置魔法函数，不用再 plt.show()
def confusion(confusion_matrix):

    L = confusion_matrix.shape[0]
    classes = []
    list(map(lambda x:classes.append(str(x)),range(L)))

    proportion = []
    for i in confusion_matrix:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)

    pshow = []
    for i in proportion:
        pt = "%.2f%%" % (i * 100)  #百分号显示准确率
        #pt = "%.2f" % (i)  #小数显示准确率
        pshow.append(pt)
    proportion = np.array(proportion).reshape(confusion_matrix.shape[0], confusion_matrix.shape[1])
    pshow = np.array(pshow).reshape(confusion_matrix.shape[0], confusion_matrix.shape[1])  # reshape(列的长度，行的长度)

    plt.figure(figsize=(10, 8))
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    plt.colorbar().ax.tick_params(labelsize=17)  # 设置右侧色标刻度大小
    #plt.colorbar().set_label(family='Times New Roman')

    tick_marks = np.arange(len(classes))  # [0, 1, 2, 3]
    plt.xticks(tick_marks, classes, fontsize=17,family='Times New Roman')
    plt.yticks(tick_marks, classes, fontsize=17,family='Times New Roman')
    ax = plt.gca()
    # 设置 横轴 刻度 标签 显示在顶部
    ax.tick_params(axis="x", top=False, labeltop=False, bottom=True, labelbottom=True)

    # thresh = confusion_matrix.max() / 2.
    # ij配对，遍历矩阵迭代器
    iters = np.reshape([[[i, j] for j in range(len(classes))] for i in range(len(classes))], (confusion_matrix.size, 2))
    for i, j in iters:
        if (i == j):
            # 仅居中显示数字
            # plt.text(j, i, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10,color='white',weight=5)

            # 同时居中显示数字和百分比
            plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=12, color='white',
                     weight=5, family='Times New Roman')  # 显示数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12, color='white',family='Times New Roman')  # 显示百分比
        else:
            # 仅居中显示数字
            # plt.text(j, i, format(confusion_matrix[i, j]),va='center',ha='center',fontsize=10)

            # 同时居中显示数字和百分比
            plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=12, family='Times New Roman')  # 显示数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12, family='Times New Roman')  # 显示百分比

    # plt.title('confusion_matrix')
    plt.ylabel('实际标签', fontsize=17,fontproperties=font_manager.FontProperties(family='SimSun'))
    plt.xlabel('预测标签', fontsize=17,fontproperties=font_manager.FontProperties(family='SimSun'))
    ax = plt.gca()
    # 设置 横轴标签 显示在顶部
    ax.xaxis.set_label_position('bottom')
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域，并且防止子图标签堆叠

    # plt.savefig(r'D:\Users\Administrator\Desktop\混淆矩阵.png', dpi=600, bbox_inches='tight')
    # plt.show()

