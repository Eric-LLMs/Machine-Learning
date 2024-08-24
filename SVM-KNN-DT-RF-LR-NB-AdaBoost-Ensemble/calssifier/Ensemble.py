# -*- coding: UTF-8 -*- 
from calssifier.common import output_data
import numpy
from sklearn import metrics

def vote_predict(fileList,filelist_test_pred,unlabeled_result_file,test_target_file):
    test_target =  numpy.load(test_target_file)
    test_data = []  #测试数据预测值 
    vote_result_test = []
    result_data = []
    vote_result = [] #投票结果 最终输出
   
    for file in fileList:
        a = open(file, "r")     
        data=a.readlines() 
        a.close()
        result_data.append(data)
 
    for file in filelist_test_pred:
        data = numpy.load(file)
        test_data.append(data)

    num = len(result_data)
     

    print ('*************************\nclassifiers voting\n*************************')

    for a0,a1,a2,a3,a4,a5,a6,a7 in  zip(result_data[0],result_data[1],result_data[2],result_data[3],result_data[4],result_data[5],result_data[6],result_data[7]) :
        a_p = 0
        a_n = 0 
        if 'negative' in a0 :
           a_n +=1
        else :
           a_p +=1

        if 'negative' in a1 :
           a_n +=1
        else :
           a_p +=1

        if 'negative' in a2 :
           a_n +=1
        else :
           a_p +=1

        if 'negative' in a3 :
           a_n +=1
        else :
           a_p +=1

        if 'negative' in a4 :
           a_n +=1
        else :
           a_p +=1

        if 'negative' in a5 :
           a_n +=1
        else :
           a_p +=1

        if 'negative' in a6 :
           a_n +=1
        else :
           a_p +=1

        if 'negative' in a7 :
           a_n +=1
        else :
           a_p +=1
        
        if a_n > a_p:
           vote_result.append(a0.replace('positive','negative'))
        elif a_n < a_p:
           vote_result.append(a0.replace('negative','positive'))
        else :  #默认准确率最高的逻辑回归
           vote_result.append(a0)
        

    for a0,a1,a2,a3,a4,a5,a6,a7 in  zip(test_data[0],test_data[1],test_data[2],test_data[3],test_data[4],test_data[5],test_data[6],test_data[7]) :
        a_p = 0
        a_n = 0 
        if 'negative' in a0 :
           a_n +=1
        else :
           a_p +=1

        if 'negative' in a1 :
           a_n +=1
        else :
           a_p +=1

        if 'negative' in a2 :
           a_n +=1
        else :
           a_p +=1

        if 'negative' in a3 :
           a_n +=1
        else :
           a_p +=1

        if 'negative' in a4 :
           a_n +=1
        else :
           a_p +=1

        if 'negative' in a5 :
           a_n +=1
        else :
           a_p +=1

        if 'negative' in a6 :
           a_n +=1
        else :
           a_p +=1

        if 'negative' in a7 :
           a_n +=1
        else :
           a_p +=1
        
        if a_n > a_p:
           vote_result_test.append('negative')
        elif a_n < a_p:
           vote_result_test.append('positive')
        else :  #默认准确率最高的逻辑回归
           if 'positive' in a0 :
             vote_result_test.append('positive')  
           if 'negative' in a0 :
             vote_result_test.append('negative')
    vote_result_test = numpy.array(vote_result_test)
    accuracy=numpy.mean(vote_result_test == test_target)
    print ("The accuracy of sc_test is %s" %accuracy)
    print(metrics.classification_report(test_target, vote_result_test))              
    output_data(unlabeled_result_file,vote_result)
    print ("The file of result is " + unlabeled_result_file)      