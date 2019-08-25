import pandas as pd
import numpy as np
from keras import models
from keras import layers
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from keras.wrappers import scikit_learn
data = pd.read_csv('/home/tianyou/文档/研究生作业.csv')

X = np.array(data.iloc[:72,1:7])
Y = np.array(data.iloc[:72,9:11])
print(X,X.shape)
X_test = np.array(data.iloc[72:,1:7])
Y_test = np.array(data.iloc[72:,9:11])
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=13)
# X = np.array(data.iloc[:75,1:7])
# Y = np.array(data.iloc[:75,9])
#
# X_test = np.array(data.iloc[75:,1:7])
# Y_test = np.array(data.iloc[75:,9])

mean = X_train.mean(axis=0)
X -= mean
std = X.std(axis=0)
X /= std
mean1 = X_val.mean(axis=0)
X_val -= mean1
std1 = X_val.std(axis=0)
X_val /= std1
X_test -= mean
X_test /= std

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(16,activation='relu',
                           input_shape=(X.shape[1],)))
    model.add(layers.Dense(16,activation='relu'))
    model.add(Dropout(0.2))
    model.add(layers.Dense(2))
    model.compile(optimizer= 'nadam',loss='mse',metrics=['mae'])
    return model
#调参代码
# k = 4
# num_val_samples = len(X) // k
# num_epochs = 30
# all_mae_histories = []
#
# for i in range(k):
#     val_data = X[i * num_val_samples:(i+1)*num_val_samples]
#     Y_targets = Y[i * num_val_samples:(i+1)*num_val_samples]
#
#     partial_train_data = np.concatenate(
#         [X[:i * num_val_samples],
#          X[(i + 1) * num_val_samples:]],
#         axis=0
#     )
#     partial_train_targets = np.concatenate(
#         [Y[:i * num_val_samples],
#          Y[(i+1) * num_val_samples:]],
#         axis=0
#     )
#
#     model = build_model()
#     history = model.fit(partial_train_data,partial_train_targets,
#               validation_data=(val_data,Y_targets),
#               epochs=num_epochs,batch_size=1,verbose=0)
#     mae_history = history.history['val_mean_absolute_error']
#     all_mae_histories.append(mae_history)
#
# average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
#
#
#
# def smooth_curve(points, factor=0.9):
#     smoothed_points = []
#     for point in points:
#         if smoothed_points:
#             previous = smoothed_points[-1]
#             smoothed_points.append(previous * factor + point * (1-factor))
#         else:
#             smoothed_points.append(point)
#     return smoothed_points
#
# smooth_mae_history = smooth_curve(average_mae_history[10:])
# plt.plot(range(1,len(smooth_mae_history)+1),smooth_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()

model = build_model()
model.fit(X,Y,epochs=15,batch_size=1,verbose=1,validation_data=(X_val,y_val))
test_mse_score,test_mae_score = model.evaluate(X_test,Y_test)
print(test_mae_score)
# seed =7
# np.random.seed(seed)
# model1 = scikit_learn.KerasClassifier(build_fn=build_model,epochs=15,batch_size=1,verbose=0)
#
# optimizer = ['sgd', 'rmsprop', 'adam', 'adagrad','nadam','adamax','adadelta']
# param_grid = dict(optimizer=optimizer)
# grid = GridSearchCV(estimator=model1, param_grid=param_grid, n_jobs=1)
# grid_result = grid.fit(X, Y)
#
# print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
#
# for mean, std, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, std, param))
