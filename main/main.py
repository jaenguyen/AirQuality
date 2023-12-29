# import libs
import pandas as pd
import Sklearn
from Math import round6
# load data
df = pd.read_csv('./resources/dataset/data_clean.csv')
# lấy cột time => mđ hiển thị dữ liệu
date = df['Date']
df = df.drop('Date', axis=1)

""" tính độ tương quan """
# tách feature and label
list_feature = list(df.columns)[:-1]
label = df.columns[len(df.columns)-1]
# tính độ tương quan giữa từng feature với label
value_label = df[label].values.tolist()
list_correlation = []
for feature in list_feature:
    val_feature = df[feature].values.tolist()
    corr = Sklearn.correlation(feature=val_feature, label=value_label)
    list_correlation.append([feature, corr])
# sắp xếp lại giá trị theo thự tự tăng dần 
list_correlation = sorted(list_correlation, key=lambda x: x[1])
# chuyển list -> df -> csv
# TODO: update convert trực tiếp sang sql
list_feature = []
list_value_corr= []
tmp = []   
for i in list_correlation:
    list_feature.append(i[0])
    tmp.append(round6(i[1]))
list_value_corr.append(tmp)
df_corr = pd.DataFrame(list(list_value_corr), columns=list_feature)
df_corr.to_csv('./resources/dataset/df_corr.csv', index=False)

""" train model và dự đoán """
# tách 2 df X, y
X = df.drop(label, axis=1)
y = df[label]
# tách dự liệu thành train và test
X_train, X_test, y_train, y_test, date_train, date_test = Sklearn.train_test_split(X, y, date, 0.3)
# khởi tạo model
model = Sklearn.LinearRegression()
# train model
model.fit(X_train, y_train)
# dự đoán 
y_pred = model.predict(X_test)
y_pred = [round6(x) for x in y_pred]
# so sánh thực tế và dự đoán
df_predict = pd.DataFrame(
    {'Date': date_test, 'Actual value': y_test, 'Predicted value': y_pred})
df_predict.to_csv('./resources/dataset/df_predict.csv', index=False)

""" tính toán các metrics """
metrics_values = []
for index in range(len(list_feature)-1):
    mse = Sklearn.metrics().mean_squared_error(y_test, y_pred)
    mae = Sklearn.metrics().mean_absolute_error(y_test, y_pred)
    r2 = model.score(X_test, y_test)
    metrics_values.append([round6(mse), round6(mae), round6(r2)])

    feature_name = list_feature[index]
    X = X.drop(feature_name, axis=1)
    # chia lại mỗi lần drop 1 feature
    X_train, X_test, y_train, y_test, date_train, date_test = Sklearn.train_test_split(X, y, date, 0.3)
    # train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
df_metrics = pd.DataFrame(metrics_values, columns=['mse', 'mae', 'r2'])
df_metrics.to_csv('./resources/dataset/df_metrics.csv', index=False)
