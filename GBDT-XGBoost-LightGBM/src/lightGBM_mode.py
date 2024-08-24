import os
import lightgbm as lgb
import pandas as pd
import math
import GBDT.config as config
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

conf = config.create_params()

model_path = os.path.join(conf.output_dir,'lgb.mdl')

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
# print(df.tail())
data = shuffle(df)
# print(data.head())
# print(data.tail())
data = shuffle(data)
# print(data.head())
# print(data.tail())
data = shuffle(data)
print(data.head())
print(data.tail())

# data.set_index('bookname',inplace=True)

data_train,data_val=train_test_split(data, test_size=0.5,random_state=1)
# num = len(data)
# data_train,data_val=data[:math.ceil(num * 0.8)],data[math.ceil(num * 0.8):]

valid_rate = 0.3
valid_num = math.floor(len(data_val) * valid_rate)
X_train = data_train.iloc[:,:-1]
y_train   = data_train.iloc[:,-1]
X_valid = data_val.iloc[:valid_num,:-1]
y_valid = data_val.iloc[:valid_num,-1]
X_test =  data_val.iloc[valid_num:,:-1]
y_test = data_val.iloc[valid_num:,-1]
df.dropna()
print(X_train)

# iris = load_iris()
# data = iris.data
# target = iris.target
# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# 创建成lgb特征的数据集格式
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

# 将参数写成字典下形式
# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',  # 设置提升类型
#     'metric': {'l2', 'auc'},  # 评估函数
#     'num_leaves': 200,  # 叶子节点数
#     'max_depth':20,
#     'objective':'binary',
#     'num_class':2,
#     'learning_rate': 0.05,  # 学习速率
#     'feature_fraction': 0.9,  # 建树的特征选择比例
#     'bagging_fraction': 0.8,  # 建树的样本采样比例
#     'bagging_freq': 50,  # k 意味着每 k 次迭代执行bagging
#     'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
# }

# lightgbm
# params = {
#     'num_leaves':150,
#     'objective':'binary',
#     'max_depth':12,
#     'learning_rate':.005,
#     'feature_fraction': 0.7,  # 建树的特征选择比例
#     'bagging_fraction': 0.8,  # 建树的样本采样比例
#     'bagging_freq': 50,  # k 意味着每 k 次迭代执行bagging
#     'verbose': 1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
#     'max_bin':200}
# params['metric'] = ['auc', 'binary_logloss']

params = {'num_leaves': 60, #结果对最终效果影响较大，越大值越好，太大会出现过拟合
          'min_data_in_leaf': 30,
          'objective': 'binary', #定义的目标函数
          'max_depth': -1,
          'learning_rate': 0.03,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,  #提取的特征比率
          "bagging_freq": 1,
          "bagging_fraction": 0.8,
          "bagging_seed": 11,
          "lambda_l1": 0.1,             #l1正则
          # 'lambda_l2': 0.001,     #l2正则
          "verbosity": -1,
          "nthread": -1,                #线程数量，-1表示全部线程，线程越多，运行的速度越快
          'metric': {'binary_logloss', 'auc'},  ##评价函数选择
          "random_state": 2019, #随机数种子，可以防止每次运行的结果不一致
          # 'device': 'gpu' ##如果安装的事gpu版本的lightgbm,可以加快运算
          }

# 训练 cv and train
gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=16)

# 保存模型到文件
gbm.save_model(model_path)

# 预测数据集
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# y_pred = gbm.predict(X_test, num_iteration=10000)
# print(y_pred)
# y_pred=[list(x).index(max(x)) for x in y_pred]
for i in range(0,len(y_pred)):
    # print(y_pred[i])
    if y_pred[i] >= 0.5:       # setting threshold to .65
        y_pred[i] = 1
    else:
        y_pred[i] = 0
# 评估模型
# print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
print('The accuracy rate of prediction is:', accuracy_score(y_test,y_pred))
auc = roc_auc_score(y_test, y_pred)
print('The auc of prediction is:', auc)
# print(gbm.feature_importance())
print(pd.DataFrame({
        'column': cols[:-1],
        'importance': gbm.feature_importance(),
    }).sort_values(by='importance',ascending = [False]))

lgb.plot_importance(gbm, max_num_features=15)  # max_features表示最多展示出前10个重要性特征，可以自行设置
plt.show()

