# -*- coding: UTF-8 -*- 
import logging
from sklearn.pipeline import Pipeline
from sklearn.linear_model import  LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy 
from sklearn import metrics
from calssifier.common import output_data, output_data_array

def logisticRegression_predict(train_file,unlabeled_file,unlabeled_data_file,train_data_file,train_target_file,test_data_file,test_target_file,unlabeled_result_file):
    train_data =  numpy.load(train_data_file)
    train_target  = numpy.load(train_target_file)
    test_data   =  numpy.load(test_data_file)
    test_target =  numpy.load(test_target_file)
    unlabeled_data =  numpy.load(unlabeled_data_file)
   
    print ('*************************\nLogisticRegression\n*************************')
    text_clf = Pipeline([('vect', CountVectorizer()), 
                ('tfidf', TfidfTransformer()), 
                ('clf', LogisticRegression()), 
                ])
    text_clf.fit(train_data, train_target)
    
    if(test_data_file=='')   :
        return  

    nb_test_predicted = text_clf.predict(test_data)
    
    import os
    test_data_pred_file = os.path.join(os.path.dirname(test_data_file),'test.file.vote.lr.npy')
    output_data_array(test_data_pred_file,nb_test_predicted)

    accuracy=numpy.mean(nb_test_predicted == test_target)
    
    print ("The accuracy of sc_test is %s" %accuracy)
    print(metrics.classification_report(test_target, nb_test_predicted))


    print ("predicting the unlable data")
    relust_list = []
    unlabeled_index = []
    unlabeled_docs = open(unlabeled_file).readlines()

    for doc in unlabeled_docs: 
         num = ''
         try:      
           word_full = ''
           word_pairs = doc.replace('\"',' ').replace('\t','').split(' ') 
           for word_pair in word_pairs:
               num =word_pairs[0] 
           unlabeled_index.append(num)
         except Exception as Error:
              logging.error(Error)

    unlabeled_predicted = text_clf.predict(unlabeled_data)

    nvs = zip(unlabeled_index,unlabeled_predicted)
    for num,result in nvs:
        relust_list.append(num+' '+ result+'\n')

    output_data(unlabeled_result_file,relust_list)

    print ("The file of result is " + unlabeled_result_file)