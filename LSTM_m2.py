import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM
from tensorflow.keras.optimizers import Adam

##### data load
df = pd.read_csv('data path')
pd.to_datetime(df['time'])
df.set_index('time', inplace = True)

##### train test split
def ts_train_test_normalize(all_data, time_steps, for_periods):
    '''
    time steps: 사용할 과거 데이터의 time steps
    for period: 예측할 미래 데이터의 time steps
    '''
    # create training and test set 
    ts_train = all_data[:38247].values
    ts_test = all_data[38247:].values 
    ts_train_len = len(ts_train)
    ts_test_len = len(ts_test)
    
    # scale the data 
    from sklearn.preprocessing import MinMaxScaler 
    sc11 = MinMaxScaler()
    ts_train_scaled = sc11.fit_transform(ts_train)
    
    # create training data of s samples and t time steps 
    X_train = [] 
    y_train = [] 
    for i in range(time_steps, ts_train_len - for_periods + 1):
        X_train.append(ts_train_scaled[i-time_steps:i, :])
        y_train.append(ts_train_scaled[i:i + for_periods, 1]) 
 
    X_train, y_train = np.array(X_train), np.array(y_train)


    # Preparing X_test
    inputs = all_data.values  # Use all columns in the DataFrame
    inputs = inputs[len(inputs)-len(ts_test)-time_steps:]
    sc22 = MinMaxScaler()
    inputs = sc22.fit_transform(inputs)

    X_test = [] 
    for i in range(time_steps, ts_test_len + time_steps - for_periods):
        X_test.append(inputs[i-time_steps:i,:])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], inputs.shape[1]))

    return X_train, y_train , X_test, sc11, sc22

X_train, y_train, X_test, sc11, sc22 = ts_train_test_normalize(df, 288, 1)

##### y_test 설정
y_test = df.loc['2020-04-26 19:20:00':, ['Rhair']]
y_test_scaled = sc22.fit_transform(y_test)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test_scaled.shape)
'''
(37959, 288, 15)
(37959, 1)
(9561, 288, 15)
(9561, 1)
'''

##### Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100.0)
])
# learning rate
learning_rate = 0.005

# training parameters
model.compile(loss=tf.keras.losses.Huber(),
              optimizer='adam',
              metrics=["mae", "mse"])
model.fit(X_train, y_train, epochs=100)

pred = model.predict(X_test)

##### results
from sklearn.metrics import mean_absolute_error, mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test_scaled, pred))
print("RMSE: %f" % (rmse))  

mse = mean_squared_error(y_test_scaled, pred)
print('mae: %f' %(mse)) 

mae = mean_absolute_error(y_test_scaled, pred)
print('mae: %f' %(mae))  

from sklearn.metrics import r2_score
r2 = r2_score(y_test_scaled, pred)
print('R2: %f' %(r2))

##### inverse transform
y_test_array = np.array(y_test_scaled) #series를 numpy로 변환
y_test_array=y_test_array.reshape(-1, 1)
y_test_inverse = sc22.inverse_transform(y_test_array)

y_pred_array = np.array(pred)
y_pred_array = y_pred_array.reshape(-1, 1)
y_pred_inverse = sc22.inverse_transform(y_pred_array)
