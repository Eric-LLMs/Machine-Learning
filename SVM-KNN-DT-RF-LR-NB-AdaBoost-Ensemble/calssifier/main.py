# -*- coding: UTF-8 -*- 
import os
from calssifier.common import SetLoger
from calssifier.SVM import *
from calssifier.NaiveBayes import *
from calssifier.KNN import *
from calssifier.LogisticRegression import logisticRegression_predict
from calssifier.DecisionTree import decisionTreeClassifier_predict
from calssifier.RandomForest import randomForestClassifier_predict
from calssifier.AdaBoost import adaBoostClassifier_predict

root_dir = 'F:\SemesterSec\dandan_machtion learning\MKLD including comments and unlabled\sc_data'
SetLoger(root_dir,'mylog.txt')

train_file = os.path.join(root_dir, 'review.train')  #训练数据文件路径
unlabeled_file = os.path.join(root_dir, 'review.unlabeled')    #预测数据文件路径
 
train_data_file = os.path.join(root_dir, 'train_data_file.npy')  #训练数据数据文件
train_target_file = os.path.join(root_dir, 'train_target_file.npy')  #训练数据标签文件
test_data_file = os.path.join(root_dir, 'test_data_file.npy')      #测试（评估）数据数据文件
test_target_file = os.path.join(root_dir, 'test_target_file.npy')  #测试（评估）数据标签文件
unlabeled_data_file = os.path.join(root_dir, 'unlabeled_data_file.npy')  #预测数据数据文件
unlabeled_result_file_ = os.path.join(root_dir, 'result.list')          #预测数据标签文件

eva_percent = 0.1  #测试（评估）数据在训练数据中百分比
eva_minIndex = 1   #抽取测试（评估）数据范围的下限
total_counts = 4000  #训练数据总个数
isRandomForEva = 'true' #isRandomForEva = 'true' 表明从测试（评估）数据中抽取做评估   isRandomForEva = 'false' 要提供测试（评估）试数据 self_test_file为测试（评估）数据的文件路径
self_test_file =  os.path.join(root_dir, 'review_test.train')       #自己提供评估模型的数据集

#predata(train_file,unlabeled_file,isRandomForEva,self_test_file,total_counts,eva_minIndex,eva_percent,train_data_file,train_target_file,test_data_file,test_target_file,unlabeled_data_file)

unlabeled_result_file = unlabeled_result_file_+'.nb'
multinomial_NaiveBayes_predict(train_file,unlabeled_file,unlabeled_data_file,train_data_file,train_target_file,test_data_file,test_target_file,unlabeled_result_file)
unlabeled_result_file = unlabeled_result_file_+'.knn'
knn_predict(train_file,unlabeled_file,unlabeled_data_file,train_data_file,train_target_file,test_data_file,test_target_file,unlabeled_result_file)
unlabeled_result_file = unlabeled_result_file_+'.svm'
svm_predict(train_file,unlabeled_file,unlabeled_data_file,train_data_file,train_target_file,test_data_file,test_target_file,unlabeled_result_file)
unlabeled_result_file = unlabeled_result_file_+'.svm.hv'
svm_predict_HashingVectorizer(train_file,unlabeled_file,unlabeled_data_file,train_data_file,train_target_file,test_data_file,test_target_file,unlabeled_result_file)
unlabeled_result_file = unlabeled_result_file_+'.lr'
logisticRegression_predict(train_file,unlabeled_file,unlabeled_data_file,train_data_file,train_target_file,test_data_file,test_target_file,unlabeled_result_file)
unlabeled_result_file = unlabeled_result_file_+'.dt'
decisionTreeClassifier_predict(train_file,unlabeled_file,unlabeled_data_file,train_data_file,train_target_file,test_data_file,test_target_file,unlabeled_result_file)
unlabeled_result_file = unlabeled_result_file_+'.rf'
randomForestClassifier_predict(train_file,unlabeled_file,unlabeled_data_file,train_data_file,train_target_file,test_data_file,test_target_file,unlabeled_result_file)
unlabeled_result_file = unlabeled_result_file_+'.ab'
adaBoostClassifier_predict(train_file,unlabeled_file,unlabeled_data_file,train_data_file,train_target_file,test_data_file,test_target_file,unlabeled_result_file)

#
#运行上面分类器完成后，注释掉，重新运行投票
#
voteFileList = []
filelist_test_pred = [] 
voteFileList.append(unlabeled_result_file_+'.lr')
voteFileList.append(unlabeled_result_file_+'.nb')
voteFileList.append(unlabeled_result_file_+'.knn')
voteFileList.append(unlabeled_result_file_+'.svm')
voteFileList.append(unlabeled_result_file_+'.svm.hv')
voteFileList.append(unlabeled_result_file_+'.dt')
voteFileList.append(unlabeled_result_file_+'.rf')
voteFileList.append(unlabeled_result_file_+'.ab')
unlabeled_result_file = unlabeled_result_file_+'.vote'

filelist_test_pred = [] 
filelist_test_pred.append(os.path.join(root_dir,'test.file.vote.lr.npy'))
filelist_test_pred.append(os.path.join(root_dir,'test.file.vote.nb.npy'))
filelist_test_pred.append(os.path.join(root_dir,'test.file.vote.knn.npy'))
filelist_test_pred.append(os.path.join(root_dir,'test.file.vote.svm.npy'))
filelist_test_pred.append(os.path.join(root_dir,'test.file.vote.svm.hv.npy'))
filelist_test_pred.append(os.path.join(root_dir,'test.file.vote.dt.npy'))
filelist_test_pred.append(os.path.join(root_dir,'test.file.vote.rf.npy'))
filelist_test_pred.append(os.path.join(root_dir,'test.file.vote.ab.npy'))

#vote_predict(voteFileList,filelist_test_pred,unlabeled_result_file,test_target_file)
 
print('ok')

