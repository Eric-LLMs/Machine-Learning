import  numpy as np
import xgboost as xgb
from sklearn.utils import shuffle
import warnings
import pandas as pd
from LevelReading.levelreading_book.training.training_features import *
from LevelReading.levelreading_book.conf.params import *

warnings.filterwarnings(action='ignore', category=UserWarning, module='matplotlib')
pd.options.mode.chained_assignment = None  # default='warn'
config = create_params_processing_book()
#线上测评数据
df_test = pd.read_csv(os.path.join(config.dir_root,config.data_test),encoding='gbk')
#训练数据
df = pd.read_csv(os.path.join(config.dir_root,config.data_training),encoding='gbk')

cols= Cols
df=df[cols]
df = df.dropna()
df_test=df_test[cols]
df_test = df_test.dropna()

data_train = shuffle(df)
data_val = shuffle(df_test)
X = data_train.iloc[:,1:-1]
y   = data_train.iloc[:,-1]
val_X = data_val.iloc[:,1:-1]
val_y = data_val.iloc[:,-1]
val_bookName =  data_val.iloc[:,0]

xgb_train = xgb.DMatrix(X, y)
xgb_val = xgb.DMatrix(val_X, val_y)

model_path = os.path.join(config.dir_root, config.model_path)

def train_model():
    # 模型参数
    params = {
        'booster': 'gbtree',
        'objective': 'reg:gamma',
        'gamma': 0.03,
        'max_depth': 4,
        'lambda': 3,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 8.5,
        'silent': 1,
        'eta': 0.02,
        'seed': 1000}
    num_rounds =200
    plst = params.items()
    model = xgb.train(plst, xgb_train, num_rounds)
    model.save_model(model_path)


def predic_online(threshold =0.5):# 预测难度与实际难度在2个年级之内认为合理，即预测值与真实值间隔2个级别之内

    bst = xgb.Booster({'nthread': 4})
    bst.load_model(model_path)
    ypred = np.array(bst.predict(xgb_val))
    dic_pre ={}
    for b,p,v in zip(val_bookName,ypred, val_y):
        if b in dic_pre.keys():
            dic_pre[b].append((p,v))
        else:
            dic_pre[b]=[(p,v)]
    book_num = len(dic_pre)
    right = []
    for k,v in dic_pre.items():
         is_right = 0
         tags = []
         for pre_g in v:
             label = pre_g[1]
             tags.append(label)
         tags = np.array(tags)
         m = tags.max()
         n = tags.min()
         max = m +  GradeThreshold[m]
         min = n - GradeThreshold[n]
         for pre_g in v:
             label = pre_g[1]
             pre_value = pre_g[0]
             if pre_value>min and pre_value<max:
                 is_right = 1
                 break
         if is_right==0:
            print('%s,%s\t预测错误'%(k, v))
         right.append(is_right)

    print('**************************************')
    print('线上准确率:%f'%(sum(right)/book_num))

def get_grad_by_score(score,threshold):
    grads = []
    for i in range(1,13):
       if i > score-threshold and i< score+threshold:
         grads.append(i)
    return grads

if __name__=='__main__':
    train_model()
    predic_online()
