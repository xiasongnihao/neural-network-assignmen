# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 10:52:11 2021

@author: xiasong
"""
import torch
import torchvision.datasets as data
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
import scipy.io
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import numpy as np
# 颜色设置
index=torch.rand(50)*279
index=index.round()
index=index.sort().values
a=[]
for i in range(len(index)-1):
      if index[i]==index[i+1]:
        a.append(i)
for i in range(len(a)):
      index = index[torch.arange(index.size(0))!=(a[i]-i)]
color = ['yellow','black','aqua','green','teal','orange','navy','pink','purple','red']
# 绘图
class GetLoader(torch.utils.data.Dataset): #读取数据集函数，可以不了解
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)
def show(v2,y):
    for i in range(len(v2)):
        plt.scatter(v2[i][0],v2[i][1],color=color[int(y[i])])
    plt.show()
def show3d(v3,y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(v3)):
        ax.scatter(v3[i][0],v3[i][1],v3[i][2],color=color[int(y[i])])
    plt.show()
 
trans = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=0, std=1)
])
data_train = scipy.io.loadmat('data_train.mat') # 读取mat文件
data_test = scipy.io.loadmat('data_test.mat') # 读取mat文件
label_train = scipy.io.loadmat('label_train.mat')
# print(data.keys())  # 查看mat文件中的所有变量
train_data=torch.tensor(data_train["data_train"])
test_data= torch.tensor(data_test["data_test"])
train_label=torch.tensor(label_train["label_train"])
kf = KFold(n_splits=6)
X,Y=train_data,train_label
loss_sum=0
acc_sum=0
km_cluster = KMeans(n_clusters=50, max_iter=1, n_init=40, \
                    init='k-means++',n_jobs=-1) 
a=torch.FloatTensor(3,5)
result = km_cluster.fit(X)
centers=result.cluster_centers_
centers=torch.tensor(centers)
centers=centers.type_as(a)
torch_data = GetLoader(X,Y)
loader_train = DataLoader(torch_data, batch_size=330, shuffle=True) 
for (x,y) in loader_train:
    # x = torch.squeeze(x) # 方法一：去掉维度为1的, 也就是以28x28来分，结果不太好
#    x = x.flatten(start_dim=2,end_dim=-1) #方法2： 压平，以1x784来分
    x = torch.squeeze(x) # 去掉维度为1的

    label=np.zeros([30,1])
    label=label+3
    y=np.vstack((y,label))
    print(x.shape)
    # pca
    pca2 = decomposition.PCA(2)
    pca3 = decomposition.PCA(3)
    # 3维
    v3 = []
    pca3.fit(x)  # sklearn的pca要求输入是（m,n）m为样本数，n为维数
    for i in range(len(index)): #将随机的点作为中心
       y[int(index[i])]=3
    temp = pca3.fit_transform(x)
    temp_cent= pca3.fit_transform(centers)
#    temp=np.vstack((temp,temp_cent))
    v3.append(temp) # 3维
    print(len(v3[0]))
    # 2维
    v2 = []
    pca2.fit(x)
    print('ssss',pca2.explained_variance_ratio_) #看每个特征可以百分之多少表达整个数据集
    temp_cent= pca2.fit_transform(centers)
    temp=pca2.fit_transform(x)
#    temp=np.vstack((temp,temp_cent))
    v2.append(temp) # 2维
    print(v2)
    # 画图
    show(v2[0],y)
    show3d(v3[0],y)
    break
pca = decomposition.PCA()
pca.fit(x)