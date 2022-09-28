import numpy as np
from matplotlib import pyplot as plt, font_manager

def plot_embedding(data, label,classes):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(5,4))
    #fig = plt.figure()

    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内`

    ax = plt.subplot(111)
    # 创建局部变量存放Plot以绘制相应的legend
    fig_leg = []
    list(map(lambda x: fig_leg.append(x), range(classes)))
    marker = ['o', '^', 'p', 'P', '*', 's', 'x', 'X', '+', 'd', 'D', '>', 'H', 'h', '<', '1', '2']

    for i in range(data.shape[0]):
        fig_leg[int(label[i])] = plt.plot(data[i, 0], data[i, 1], linestyle='', marker=marker[int(label[i])],
                                          markersize=9, color=plt.cm.tab20(label[i] / 20.))
    my_font = font_manager.FontProperties(family='Times New Roman',size=8)

    hand = list(map(lambda x: x[0], fig_leg))
    plt.legend(loc='right', ncol=1, frameon=True, labelspacing=0.8, columnspacing=0.4, handletextpad=0.4,
               prop=my_font, handlelength=1, bbox_to_anchor=(1.12, 0.5),
               handles=hand, labels=list(range(classes)))

    # plt.xticks(fontproperties=font_manager.FontProperties(family='Times New Roman',size=8))
    # plt.yticks(fontproperties=font_manager.FontProperties(family='Times New Roman',size=8))
    plt.xticks([])
    plt.yticks([])
    plt.xlim([data[:,0].min()-0.05,data[:,0].max()+0.05])
    plt.ylim([data[:,1].min()-0.05,data[:,1].max()+0.05])

    return fig