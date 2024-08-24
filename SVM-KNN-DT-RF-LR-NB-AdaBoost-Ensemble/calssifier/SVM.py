# -*- coding: UTF-8 -*- 
import numpy
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn import metrics
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from calssifier.common import output_data, output_data_array


def svm_predict(train_file,unlabeled_file,unlabeled_data_file,train_data_file,train_target_file,test_data_file,test_target_file,unlabeled_result_file):
    train_data =  numpy.load(train_data_file)
    train_target  = numpy.load(train_target_file)
    test_data   =  numpy.load(test_data_file)
    test_target =  numpy.load(test_target_file)
    unlabeled_data =  numpy.load(unlabeled_data_file)

    print ('*************************\nSVM\n*************************')
    text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),])
    
    text_clf.fit(train_data, train_target)
    
    svm_test_predicted = text_clf.predict(test_data)
    
    import os
    test_data_vote_file = os.path.join(os.path.dirname(test_data_file),'test.file.vote.svm.npy')
    output_data_array(test_data_vote_file,svm_test_predicted)


    accuracy=numpy.mean(svm_test_predicted == test_target)
     
    print ("The accuracy of sc_test is %s" %accuracy)
   
    print(metrics.classification_report(test_target, svm_test_predicted))

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

    #少量特征
def svm_predict_HashingVectorizer(train_file,unlabeled_file,unlabeled_data_file,train_data_file,train_target_file,test_data_file,test_target_file,unlabeled_result_file):
    train_data =  numpy.load(train_data_file)
    train_target  = numpy.load(train_target_file)
    test_data   =  numpy.load(test_data_file)
    test_target =  numpy.load(test_target_file)
    unlabeled_data =  numpy.load(unlabeled_data_file)

    print ('*************************\nHashingVectorizer\n*************************')
    text_clf = Pipeline([('vect', HashingVectorizer(stop_words = 'english',non_negative = True,  
                               n_features = 10000)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),])
    
    text_clf.fit(train_data, train_target)
    
    SGD_test_predicted = text_clf.predict(test_data)
    
    import os
    test_data_vote_file = os.path.join(os.path.dirname(test_data_file),'test.file.vote.svm.hv.npy')
    output_data_array(test_data_vote_file,SGD_test_predicted)

    accuracy=numpy.mean(SGD_test_predicted == test_target)
    
    print ("The accuracy of sc_test is %s" %accuracy)
   
    print(metrics.classification_report(test_target, SGD_test_predicted))

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