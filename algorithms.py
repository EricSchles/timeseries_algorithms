import statistics as stat
from sklearn import svm, linear_model

def moving_median(df, sliding_window=1):
    medians = []
    terminal_condition = False
    start = 0
    end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while end != len(data):
        medians.append(stat.median(data[start:end]))
        start += 1
        end += 1
    return medians

def moving_median_grouped(df, sliding_window=1):
    medians = []
    terminal_condition = False
    start = 0
    end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while end != len(data):
        medians.append(stat.median_grouped(data[start:end]))
        start += 1
        end += 1
    return medians

def moving_median_high(df, sliding_window=1):
    medians = []
    terminal_condition = False
    start = 0
    end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while end != len(data):
        medians.append(stat.median_high(data[start:end]))
        start += 1
        end += 1
    return medians

def moving_median_low(df, sliding_window=1):
    medians = []
    terminal_condition = False
    start = 0
    end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while end != len(data):
        medians.append(stat.median_low(data[start:end]))
        start += 1
        end += 1
    return medians

def moving_average(df, sliding_window=1):
    means = []
    terminal_condition = False
    start = 0
    end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while end != len(data):
        means.append(stat.mean(data[start:end]))
        start += 1
        end += 1
    return means

def svm_autoregression(df, sliding_window=1):
    predictions = []
    terminal_condition = False
    x_start = 0
    x_end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while x_end != len(data)-1:
        Y = data[x_end+1]
        X = data[x_start:x_end]
        regressor = smv.SVR()
        regressor.fit(X,Y)
        predictions.append(regressor.predict(X))
        x_start += 1
        x_end += 1
    return predictions

def ridge_autoregression(df, sliding_window=1):
    predictions = []
    terminal_condition = False
    x_start = 0
    x_end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while x_end != len(data)-1:
        Y = data[x_end+1]
        X = data[x_start:x_end]
        regressor = linear_model.Ridge()
        regressor.fit(X,Y)
        predictions.append(regressor.predict(X))
    return predictions

def lasso_autoregression(df, sliding_window=1):
    predictions = []
    terminal_condition = False
    x_start = 0
    x_end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while x_end != len(data)-1:
        Y = data[x_end+1]
        X = data[x_start:x_end]
        regressor = linear_model.Lasso()
        regressor.fit(X,Y)
        predictions.append(regressor.predict(X))
    return predictions

def elastic_net_autoregression(df, sliding_window=1):
    predictions = []
    terminal_condition = False
    x_start = 0
    x_end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while x_end != len(data)-1:
        Y = data[x_end+1]
        X = data[x_start:x_end]
        regressor = linear_model.ElasticNet()
        regressor.fit(X,Y)
        predictions.append(regressor.predict(X))
    return predictions

def orthogonal_matching_pursuit_cv_autoregression(df, sliding_window=1):
    predictions = []
    terminal_condition = False
    x_start = 0
    x_end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while x_end != len(data)-1:
        Y = data[x_end+1]
        X = data[x_start:x_end]
        regressor = linear_model.OrthogonalMatchingPursuitCV()
        regressor.fit(X,Y)
        predictions.append(regressor.predict(X))
    return predictions

def bayesian_ridge_autoregression(df, sliding_window=1):
    predictions = []
    terminal_condition = False
    x_start = 0
    x_end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while x_end != len(data)-1:
        Y = data[x_end+1]
        X = data[x_start:x_end]
        regressor = linear_model.BayesianRidge()
        regressor.fit(X,Y)
        predictions.append(regressor.predict(X))
    return predictions

def automatic_relevance_determination_regression(df, sliding_window=1):
    predictions = []
    terminal_condition = False
    x_start = 0
    x_end = sliding_window
    data = [df.ix[ind] for ind in df.index]
    while x_end != len(data)-1:
        Y = data[x_end+1]
        X = data[x_start:x_end]
        regressor = linear_model.ARDRegression()
        regressor.fit(X,Y)
        predictions.append(regressor.predict(X))
    return predictions
