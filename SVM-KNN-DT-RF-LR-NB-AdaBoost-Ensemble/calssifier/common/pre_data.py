# -*- coding: UTF-8 -*- 
import logging
from calssifier.common.post_data import *

def predata(train_file,unlabeled_file,isRandomForEva,self_test_file,total_counts,eva_minIndex,eva_percent,train_data_file,train_target_file,test_data_file,test_target_file,unlabeled_data_file):
    train_data =  []
    train_target  = []
    test_data   = []
    test_target = []
    unlabeled_data = []
    unlabeled_docs = open(unlabeled_file).readlines()
    for doc in unlabeled_docs: 
         num = ''
         try:      
           word_full = ''
           #label = doc.split(' ')[1].replace(':','').replace('\n','')
           word_pairs = doc.replace('\"',' ').replace('\t','').split(' ') 
           for word_pair in word_pairs:
               num =word_pairs[0] 
               word_pair_arry = word_pair.split(':')
               if len(word_pair_arry)< 2:
                    continue
               else:                                      
                   for i in range(0,int(word_pair_arry[1])) :
                     #.replace('_',' ').replace('\"',' ').replace('-',' ').replace('\t','').split(' ') 
                     word_full += word_pair_arry[0]+' '
           unlabeled_data.append(num+'\t'+word_full)
         except Exception as Error:
              logging.error(Error)

    docs = open(train_file).readlines()
    doc_index = 0
    if(isRandomForEva == 'true' ) :
        testIndexs = generat_randow(eva_minIndex,total_counts,eva_percent)
        for doc in docs: 
         try:      
           word_full = ''
           doc_index +=1
           label = doc.split(' #label#')[1].replace(':','').replace('\n','')
           #word_pairs = doc.replace('_',' ').replace('\"',' ').replace('-',' ').replace('\t','').split(' #label#')[0].split(' ') 
           word_pairs = doc.replace('\t','').split(' #label#')[0].split(' ') 
           for word_pair in word_pairs:
               word_pair_arry = word_pair.split(':')
               if len(word_pair_arry)< 2:
                    continue
               else:                                      
                   for i in range(0,int(word_pair_arry[1])) :
                     word_full += word_pair_arry[0]+' '
           if doc_index in testIndexs:
                 test_data.append(word_full)
                 test_target.append(label)
           else:
              train_data.append(word_full)
              train_target.append(label)
         except Exception as Error:
              logging.error(Error)
    else :
        for doc in docs: 
             try:      
               word_full = ''
               doc_index +=1
               label = doc.split(' #label#')[1].replace(':','').replace('\n','')
               #word_pairs = doc.replace('_',' ').replace('\"',' ').replace('-',' ').replace('\t','').split(' #label#')[0].split(' ') 
               word_pairs = doc.replace('\t','').split(' #label#')[0].split(' ')
               for word_pair in word_pairs:
                   word_pair_arry = word_pair.split(':')
                   if len(word_pair_arry)< 2:
                        continue
                   else:                                      
                       for i in range(0,int(word_pair_arry[1])) :
                         word_full += word_pair_arry[0]+' '
               train_data.append(word_full)
               train_target.append(label)
             except Exception as Error:
                  logging.error(Error)
        self_test_docs = open(self_test_file).readlines()
        for doc in self_test_docs: 
             try:      
               word_full = ''
               doc_index +=1
               label = doc.split(' #label#')[1].replace(':','').replace('\n','')
               #word_pairs = doc.replace('_',' ').replace('\"',' ').replace('-',' ').replace('\t','').split(' #label#')[0].split(' ') 
               word_pairs = doc.replace('\t','').split(' #label#')[0].split(' ')
               for word_pair in word_pairs:
                   word_pair_arry = word_pair.split(':')
                   if len(word_pair_arry)< 2:
                        continue
                   else:                                      
                       for i in range(0,int(word_pair_arry[1])) :
                         word_full += word_pair_arry[0]+' '
               test_data.append(word_full)
               test_target.append(label)
             except Exception as Error:
                  logging.error(Error)
    output_data_array(train_data_file,train_data)
    output_data_array(train_target_file,train_target)
    output_data_array(test_data_file,test_data)
    output_data_array(test_target_file,test_target)
    output_data_array(unlabeled_data_file,unlabeled_data)
