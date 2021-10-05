# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 11:45:55 2021

@author: xiasong
"""

import torch, random
import torch.nn as nn
import torch.optim as optim
import scipy.io
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

index=torch.rand(100)*279
index=index.round()
index=index.sort().values 
#torch.manual_seed(42)
num_clusters = 130
km_cluster = KMeans(n_clusters=num_clusters, max_iter=300, n_init=40, \
                    init='k-means++',n_jobs=-1) 
class RBFN(nn.Module):
    """
    以高斯核作为径向基函数
    """
    def __init__(self, centers, n_out=3):
        """
        :param centers: shape=[center_num,data_dim]
        :param n_out:
        """
        super(RBFN, self).__init__()
        self.n_out = n_out
        self.num_centers = centers.size(0) # 隐层节点的个数
        self.dim_centure = centers.size(1) # 
        self.centers = nn.Parameter(centers)
        # self.beta = nn.Parameter(torch.ones(1, self.num_centers), requires_grad=True)
        self.beta = torch.ones(1, self.num_centers)*0.2
        # 对线性层的输入节点数目进行了修改
        self.linear = nn.Linear(self.num_centers+self.dim_centure, self.n_out, bias=True)
        self.Sigmoid= nn.Sigmoid()
        self.initialize_weights()# 创建对象时自动执行
 
 
    def kernel_fun(self, batches):
        n_input = batches.size(0)  # number of inputs
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)
        B = batches.view(n_input, -1).unsqueeze(1).repeat(1, self.num_centers, 1)
        C = torch.exp(-self.beta.mul((A - B).pow(2).sum(2, keepdim=False)))
        return C
 
    def forward(self, batches):
        radial_val = self.kernel_fun(batches)
        class_score = self.linear(torch.cat([batches, radial_val], dim=1))
      #  class_score= self.Sigmoid(class_score)
        return class_score
 
    def initialize_weights(self, ):
        """
        网络权重初始化
        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
 
    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(self)
        print('Total number of parameters: %d' % num_params)
 
# centers = torch.rand((5,8))
# rbf_net = RBFN(centers)
# rbf_net.print_network()
# rbf_net.initialize_weights()
 
 
if __name__ =="__main__":
    data_train = scipy.io.loadmat('data_train.mat') # 读取mat文件
    data_test = scipy.io.loadmat('data_test.mat') # 读取mat文件
    label_train = scipy.io.loadmat('label_train.mat')
# print(data.keys())  # 查看mat文件中的所有变量
    train_data=torch.tensor(data_train["data_train"])
    test_data= torch.tensor(data_test["data_test"])
    train_label=torch.tensor(label_train["label_train"])
#    val_data=torch.tensor(train_data[0:50,:])
#    val_label=torch.tensor(train_label[0:50,:])
#    train_data=torch.tensor(train_data[50:330])
#    train_label=torch.tensor(train_label[50:330])
#    data = torch.tensor([[0.25, 0.75], [0.75,0.75], [0.25,0.5], [0.5,0.5],[0.75,0.5],
#                         [0.25,0.25],[0.75,0.25],[0.5,0.125],[0.75,0.125]], dtype=torch.float32)
#    label = torch.tensor([[-1,1,-1],[1,-1,-1],[-1,-1,1],[-1,-1,1],[-1,-1,1],
#                          [1,-1,-1],[-1,1,-1],[-1,1,-1],[1,-1,-1]], dtype=torch.float32)
#    a=[]
#    for i in range(len(index)-1):
#      if index[i]==index[i+1]:
#        a.append(i)
#    for i in range(len(a)):
#      index = index[torch.arange(index.size(0))!=(a[i]-i)]
    kf = KFold(n_splits=6)
    X,Y=train_data,train_label
    loss_sum=0
    acc_sum=0
    for train_index, test_index in kf.split(train_data):
      print("Train:", train_index, "Validation:",test_index)
      data, val_data = X[train_index], X[test_index]
      label, val_label = Y[train_index], Y[test_index]
#      data=train_data
#      label=train_label
      a=torch.FloatTensor(3,5)
      data=data.type_as(a)
      label=label.type_as(a)
      print(data.size())
#    centers = data[0:len(index),:]
#    for i in range(len(index)):
#        centers[i,:]=data[int(index[i])]
      result = km_cluster.fit(data)
      centers=result.cluster_centers_
      centers=torch.tensor(centers)
      centers=centers.type_as(a)
      rbf = RBFN(centers,1)
      params = rbf.parameters()
      loss_fn = torch.nn.MSELoss()
      optimizer = torch.optim.SGD(params,lr=0.1,momentum=0.9)
      val_data=val_data.type_as(a)
      for i in range(400):
          optimizer.zero_grad()
 
          y = rbf.forward(data)
          y_test = rbf.forward(val_data)
          loss = loss_fn(y,label)
          loss_test=loss_fn(y_test,val_label)
          loss.backward()
          optimizer.step()
          if i%20 ==0:
             print(i,"\t",loss.data,"\t",loss_test.data)
      loss_sum= loss_sum+loss_test
    # 加载使用
    
      test_data=test_data.type_as(a)
      y = rbf.forward(val_data)
      count=0
      for i in range(len(y)):
          if (y[i,0]-val_label[i])<1:
              count=count+1
      acc=count/len(y)
      acc_sum=acc_sum+acc
#    print(y.data)
#    print(val_label.data)
    print(acc_sum/6)
    print(loss_sum/6)