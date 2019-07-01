import os
import sys

import time

import pandas as pd
import xgboost as xgb
import numpy as np
import GBDT.config as config
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

conf = config.create_params()

model_path = os.path.join(conf.output_dir,'model.xgb.mdl')

df = pd.read_csv(os.path.join(conf.data_dir,'corpus_danche.csv'),encoding='gbk')
#不含句子复杂度
# cols = ['dwd_ord_bike_ride_info_openlock_time','dwd_ord_bike_ride_info_ride_time','dwd_ord_bike_ride_info_no_discount_cost','dwd_ord_bike_ride_info_discount_type','dwd_ord_bike_ride_info_ride_distance','dwd_ord_bike_ride_info_is_in_insurance','dwd_ord_bike_ride_info_create_channel','dwd_ord_bike_ride_info_end_channel','dwd_ord_bike_ride_info_account_type','dwd_ord_bike_ride_info_system_code','dwd_ord_bike_ride_info_lock_out_service_area','dwd_ord_bike_ride_info_out_of_sa_penalty','dwd_ord_bike_ride_info_penalty_operate_money','dwd_ord_bike_ride_info_in_forbidden_penalty','dwd_ord_bike_ride_info_in_forbidden_creditscore','dwd_ord_bike_ride_info_in_forbidden_operate_money','dwd_ord_bike_ride_info_open_lock_error_type','dwd_ord_bike_ride_info_in_forbidden_type','dwd_ord_bike_ride_info_ride_status','dwd_ord_bike_ride_info_start_type','dwd_ord_bike_ride_info_end_type','dwd_ord_bike_ride_info_start_over_region','dwd_ord_bike_ride_info_end_over_region','dwd_ord_bike_ride_info_start_over_bigregion','dwd_ord_bike_ride_info_end_over_bigregion','dwd_ord_bike_ride_info_pay_status','lable']
# cols = ['dwd_ord_bike_ride_info_openlock_time','dwd_ord_bike_ride_info_ride_time','dwd_ord_bike_ride_info_no_discount_cost','dwd_ord_bike_ride_info_discount_type','dwd_ord_bike_ride_info_ride_distance','dwd_ord_bike_ride_info_is_in_insurance','dwd_ord_bike_ride_info_create_channel','dwd_ord_bike_ride_info_end_channel','dwd_ord_bike_ride_info_account_type','dwd_ord_bike_ride_info_system_code','dwd_ord_bike_ride_info_lock_out_service_area','dwd_ord_bike_ride_info_out_of_sa_penalty','dwd_ord_bike_ride_info_in_forbidden_penalty','dwd_ord_bike_ride_info_in_forbidden_creditscore','dwd_ord_bike_ride_info_open_lock_error_type','dwd_ord_bike_ride_info_in_forbidden_type','dwd_ord_bike_ride_info_ride_status','dwd_ord_bike_ride_info_start_type','dwd_ord_bike_ride_info_end_type','dwd_ord_bike_ride_info_start_over_region','dwd_ord_bike_ride_info_end_over_region','dwd_ord_bike_ride_info_start_over_bigregion','dwd_ord_bike_ride_info_end_over_bigregion','dwd_ord_bike_ride_info_pay_status','lable']
# cols = ['dwd_ord_bike_ride_info_openlock_time','dwd_ord_bike_ride_info_ride_time','dwd_ord_bike_ride_info_ride_distance','dwd_ord_bike_ride_info_is_in_insurance','dwd_ord_bike_ride_info_create_channel','dwd_ord_bike_ride_info_account_type','dwd_ord_bike_ride_info_system_code','dwd_ord_bike_ride_info_lock_out_service_area','dwd_ord_bike_ride_info_out_of_sa_penalty','dwd_ord_bike_ride_info_in_forbidden_penalty','dwd_ord_bike_ride_info_in_forbidden_creditscore','dwd_ord_bike_ride_info_in_forbidden_type','dwd_ord_bike_ride_info_ride_status','dwd_ord_bike_ride_info_start_type','dwd_ord_bike_ride_info_end_type','dwd_ord_bike_ride_info_start_over_region','dwd_ord_bike_ride_info_end_over_region','dwd_ord_bike_ride_info_start_over_bigregion','dwd_ord_bike_ride_info_end_over_bigregion','dwd_ord_bike_ride_info_pay_status','lable']
cols = ['dwd_ord_bike_ride_info_ride_time','dwd_ord_bike_ride_info_ride_distance','dwd_ord_bike_ride_info_openlock_time','dwd_ord_bike_ride_info_pay_status','dwd_ord_bike_ride_info_system_code','dwd_ord_bike_ride_info_end_type','dwd_ord_bike_ride_info_start_type','dwd_ord_bike_ride_info_ride_status','dwd_ord_bike_ride_info_create_channel','lable']
# cols = ['dwd_ord_bike_ride_info_ride_time','dwd_ord_bike_ride_info_ride_distance','dwd_ord_bike_ride_info_openlock_time','dwd_ord_bike_ride_info_pay_status','lable']

df=df[cols]
# df = df.dropna()
# df.fillna(-1000)
# print(df.head(-1000))
# print(df.tail())
data = shuffle(df)
data = shuffle(data)
# print(data.head())
# print(data.tail())

# data.set_index('bookname',inplace=True)

data_train,data_val=train_test_split(data, test_size=0.5,random_state=1)
# num = len(data)
# data_train,data_val=data[:math.ceil(num * 0.8)],data[math.ceil(num * 0.8):]

X_train = data_train.iloc[:,:-1]
y_train   = data_train.iloc[:,-1]
X_test = data_val.iloc[:,:-1]
y_test = data_val.iloc[:,-1]
df.dropna()
print(X_train)


ds_train = xgb.DMatrix(X_train, y_train)
ds_val = xgb.DMatrix(X_test, y_test)


def train_model():
    param = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.1,
        'max_depth': 200,
        'lambda': 1,
        'subsample': 0.75,
        'colsample_bytree': 0.7,
        'min_child_weight': 1.5,
        'silent': 0,
        'nthread': 32,
        'eta': 0.02,
        'seed': 1}

    num_rounds = 36
    plst = param.items()
    model = xgb.train(plst, ds_train, num_rounds)
    model.save_model(model_path)
    y_pred = model.predict(xgb.DMatrix(X_test))
    y_label = np.array(y_test).reshape(1, -1)[0]

    for i in range(0, len(y_pred)):
        # print(y_pred[i])
        if y_pred[i] >= 0.5:  # setting threshold to .65
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    print('The accuracy rate of prediction is:', accuracy_score(y_test, y_pred))
    auc = roc_auc_score(y_test, y_pred)
    print('The auc of prediction is:', auc)
    # train_pred = model.predict(xgb.DMatrix(val_X))
    # predictions = [round(value) for value in y_pred]
    # for i, j in zip(val_y, y_pred):
    #     print(i, j)

    # 评估预测结果
    # accuracy = accuracy_score(val_y, predictions)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))


if __name__=='__main__':
    # print("train_xgboost start \t" + time.strftime('%Y.%m.%d.%H:%M:%S', time.localtime(time.time())))
    train_model()
    # print("train_xgboost end \t" + time.strftime('%Y.%m.%d.%H:%M:%S', time.localtime(time.time())))