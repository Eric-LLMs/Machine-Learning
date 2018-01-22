import os,sys
from collections import namedtuple

pwd = os.getcwd()
root = father_path=os.path.abspath(os.path.dirname(pwd)+os.path.sep+"./run_dir")

print("数据模型目录：%s"%root)
params_processing = namedtuple(
  "Params",
  [
    "dir_root",
    'corpus_levelreading_train',
    'corpus_levelreading_test',
    "data_training",
    "data_test",
    "model_path"
  ])

def create_params_processing_book():
  return params_processing(
      dir_root=root,
      corpus_levelreading_train = os.path.join(root,'corpus_book_all'),
      corpus_levelreading_test=os.path.join(root, 'corpus_book_online_150'),
      data_training=os.path.join(root,'data_train.csv'),
      data_test=os.path.join('data_test_online_150.csv'),
      model_path=os.path.join('XGBClassifier.model')
  )

# def create_params_processing_book_fc_dnn():
#     return params_processing(
#         dir_root=root,
#         corpus_levelreading_train=os.path.join(root, 'corpus_book_all_v1'),
#         corpus_levelreading_test=os.path.join(root, 'corpus_book_online_150_v1'),
#         data_training=os.path.join(root, 'data_book_levelreading_train_v1.csv'),
#         data_test=os.path.join('data_book_levelreading_online_150_v1.csv'),
#         model_path=os.path.join('XGBClassifier.model')
#     )


