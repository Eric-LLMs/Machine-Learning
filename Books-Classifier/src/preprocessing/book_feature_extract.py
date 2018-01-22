import os,sys
from LevelReading.levelreading_book.training.training_features import *
from LevelReading.levelreading_book.conf.params import *

config = create_params_processing_book()
book_features_values = book_features_values()

#书特征值提取
def book_feature_extract(data_num):
    op = open(os.path.join(config.dir_root, config.data_training), 'w', encoding='UTF-8')
    index = 0
    for line in open(os.path.join(config.dir_root, config.corpus_levelreading_train), 'r', encoding='UTF-8'):
        if index==0 :
            op.write(line.replace('\n','') + '\n')
            index+=1
            op.flush()
            continue
        line = line.replace('\n','')
        try:
            info = line.replace('\n', '').strip( '\t' ).split('\t')[1:]
            train_line =line.replace('\n', '').strip( '\t' ).split('\t')[0]+'\t'
            for i in range(len(info)-1):
                feature_name = book_features_values.get_features_name(i)
                feature_dic = book_features_values.get_feature_values(feature_name)
                value = feature_dic[info[i]]
                train_line+= str(value)+'\t'
                # print(train_line)
            labels = info[len(info)-1].replace('\t','').replace('"','').replace(' ','') #label 一列格式  " 1,2"
            for label in labels.split(','):
                if label=='':
                    continue
                op.write(train_line+str(label)+ '\n')
                op.flush()
            index += 1
            print(float(index / data_num))
        except Exception as e:
            ope = open(os.path.join(config.dir_root, 'book_feature_extract_error'), 'a', encoding='UTF-8')
            ope.write(line.replace('\t', '') + '\n')
            ope.close()
    op.close()

if __name__=='__main__':
    # data_num = 22000
    # book_feature_extract(data_num)
    pass