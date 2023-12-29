import numpy as np
from numpy.linalg import inv

# tính độ tương quan
def correlation(feature, label):
    mean_feature = sum(feature) / float(len(feature))
    mean_label = sum(label) / float(len(label))
    sub_feature = [float(i) - mean_feature for i in feature]
    sub_label = [float(i) - mean_label for i in label]
    numerator = sum([sub_feature[i] * sub_label[i]
                    for i in range(len(sub_feature))])
    std_deviation_feature = sum(
        [sub_feature[i] ** 2.0 for i in range(len(sub_feature))])
    std_deviation_y = sum([sub_label[i] ** 2.0 for i in range(len(sub_label))])
    denominator = (std_deviation_feature * std_deviation_y) ** 0.5
    cor = numerator / denominator
    return cor

# chia data => train và test
def train_test_split(X, y, date, test_size):
    date = date.to_numpy()
    X = X.to_numpy()
    y = y.to_numpy()
    total_rows = X.shape[0]
    train_size = int(total_rows*test_size)
    return X[train_size:],  X[:train_size], y[train_size:], y[:train_size], date[train_size:], date[:train_size]

# lớp LinearRegression
class LinearRegression:
    # khởi tạo
    def __init__(self):
        pass

    # train model
    def fit(self, X_train, y_train):
        ## B = (XtX)-1(XtY)
        X = np.insert(X_train, 0, np.array(1), axis=1)
        Xt = X.T
        XtX = np.dot(Xt, X)
        XtXinv = inv(XtX)
        XtY = np.dot(Xt, y_train)
        B = np.dot(XtXinv, XtY)
        self.intercept_ = B[0]
        self.coef_ = B[1:]

    # dự đoán
    def predict(self, X_test):
        y_pred = np.dot(X_test, self.coef_)
        for i in range(y_pred.shape[0]):
            y_pred[i] += self.intercept_
        return y_pred

    # tính độ chính xác
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return metrics().r2_score(y_test, y_pred)

# lớp metrics
class metrics:
    # khởi tạo
    def __init__(self):
        pass

    # sai số toàn phương trung bình (yi-y)/n
    def mean_squared_error(self, y_test, y_pred):
        test_size = len(y_test)
        result = 0
        for i in range(test_size):
            result += pow((y_test[i]-y_pred[i]),2)
        result /= test_size
        return result

    # sai số toàn phương tuyệt đối |yi-y|/n
    def mean_absolute_error(self, y_test, y_pred):
        test_size = len(y_test)
        result = 0
        for i in range(test_size):
            result += abs(y_test[i]-y_pred[i])
        result /= test_size
        return result
        
    # hệ số xác định độ chặt chẽ 
    def r2_score(self, y_test, y_pred):
        ESS = 0
        TSS = 0
        mean_y_test = y_test.mean()
        test_size = len(y_test)
        for i in range(test_size):
            ESS += pow(y_test[i] - y_pred[i], 2)
        for i in range(test_size):
            TSS += pow(y_test[i] - mean_y_test, 2)
        result =  1 - ESS / TSS
        return result