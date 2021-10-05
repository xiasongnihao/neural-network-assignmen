# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 20:59:06 2021

@author: xiasong
"""
from sklearn.svm import SVC
clf = SVC(kernel='rbf',C=0.4)
clf.fit(data, label_svm)
y=clf.predict(val_data)
count=0
for i in range(len(y)):
    if (y[i]-val_label[i])<1:
        count=count+1
acc=count/len(y)
acc_sum=0
acc_train=0
for train_index, test_index in kf.split(train_data):
#      print("Train:", train_index, "Validation:",test_index)
      data, val_data = X[train_index], X[test_index]
      label, val_label = Y[train_index], Y[test_index]
#      data=train_data
#      label=train_label
      a=torch.FloatTensor(3,5)
      data=data.type_as(a)
      label=label.type_as(a)
      label_svm=label.reshape(len(label))
      clf = SVC(kernel='rbf',C=0.4,gamma=1)
      clf.fit(data, label_svm)
      y=clf.predict(val_data)
      count=0
      for i in range(len(y)):
        if (y[i]-val_label[i])<1:
           count=count+1
      acc=count/len(y)
      acc_sum=acc_sum+acc
      y=clf.predict(data)
      count=0
      for i in range(len(y)):
        if (y[i]-label[i])<1:
           count=count+1
      acc=count/len(y)
      acc_train=acc_train+acc
#      print(acc)

print(acc_sum/6)
print(acc_train/6)