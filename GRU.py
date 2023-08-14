import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# data load
df = pd.read_csv('file_path.csv')
pd.to_datetime(df['time'])
df.set_index('time', inplace = True)

# train test split
from sklearn.model_selection import train_test_split
def split_data(df, target_col, test_size=0.4, val_size=0.5, random_state=2021):
    feature_columns = list(df.columns.difference([target_col]))
    X = df[feature_columns]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, X_val, y_val

target_col = "Tair"  # 타겟 설정
X_train, X_test, y_train, y_test, X_val, y_val = split_data(df, target_col)

# scaling
scaler11 = MinMaxScaler()
scaler22 = MinMaxScaler()
X_train = scaler11.fit_transform(X_train)
X_test = scaler11.fit_transform(X_test)
X_val = scaler11.fit_transform(X_val)

y_train_array = np.array(y_train)
y_train_reshape = y_train_array.reshape(-1, 1)
y_train = scaler22.fit_transform(y_train_reshape)

y_test_array = np.array(y_test)
y_test_reshape = y_test_array.reshape(-1, 1)
y_test = scaler22.fit_transform(y_test_reshape)

y_val_array = np.array(y_val)
y_val_reshape = y_val_array.reshape(-1, 1)
y_val =scaler22.fit_transform(y_val_reshape)

# window size 설정
WINDOW_SIZE = 8640
BATCH_SIZE = 32

def windowed_dataset(x, y, window_size, batch_size, shuffle):
    # X 값 window dataset 구성
    ds_x = tf.data.Dataset.from_tensor_slices(x)
    ds_x = ds_x.window(WINDOW_SIZE, shift=1, stride=1, drop_remainder=True)
    ds_x = ds_x.flat_map(lambda x: x.batch(WINDOW_SIZE))

    #y 값 추가
    ds_y = tf.data.Dataset.from_tensor_slices(y[WINDOW_SIZE:])
    ds = tf.data.Dataset.zip((ds_x, ds_y))
    if shuffle:
        ds = ds.shuffle(1000)
    return ds.batch(batch_size).prefetch(1)

import tensorflow as tf
train_data = windowed_dataset(X_train, y_train, WINDOW_SIZE, BATCH_SIZE, True)
test_data = windowed_dataset(X_test, y_test, WINDOW_SIZE, BATCH_SIZE, False)

# 데이터 셋의 구성을 확인
for data in train_data.take(1):
    print(f'데이터셋(X) 구성(batch size, window size, feature 갯수): {data[0].shape}')
    print(f'데이터셋(y) 구성(batch size, window size, feature 갯수): {data[1].shape}')

#################
##### Model #####
#################
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

## input size 확인
model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((1, 14), input_shape=(14,)),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.GRU(64, return_sequences=True),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100.0)
])

# learning rate
learning_rate = 0.005

# training parameters
model.compile(loss=tf.keras.losses.Huber(),
              optimizer='adam',
              metrics=["mae"])

# Train
model.fit(X_train, y_train, epochs=100)
model.evaluate(X_test, y_test)
pred = model.predict(X_test)

# results
mse = mean_squared_error(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print('TEMP MSE: {:.4f}'.format(mse)) 
print('TEMP MAE: {:.4f}'.format(mae)) 
print('TEMP RMSE: {:.4f}'.format(rmse)) 
print('TEMP R2: {:.4f}'.format(r2))
