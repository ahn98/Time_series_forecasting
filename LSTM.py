from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# train/test split
feature_columns = list(gcwactu3.columns.difference(['Tair'])) # target을 제외한 모든 행
X = gcwactu3[feature_columns] # 설명변수
y = gcwactu3['Tair'] # target 설정

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state = 42)
#y_test의 'time' index를 사용하기 위해 내보냄
#y_test.to_csv('y_test.csv')

#%%
# Define the window size
window_size = 8640  # input data = 30일

# Reshape
X_train_windowed = []
for i in range(window_size, len(X_train)):
    X_train_windowed.append(X_train.iloc[i - window_size:i, :].values)
X_train_windowed = np.array(X_train_windowed)

X_test_windowed = []
for i in range(window_size, len(X_test)):
    X_test_windowed.append(X_test.iloc[i - window_size:i, :].values)
X_test_windowed = np.array(X_test_windowed)

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


# %%
# Build the model (양방향 모델/과거와 미래의 정보를 모두 활용한다. )
# <-> 단방향은 tf.keras.layers.LSTM(32)
# Lambda layer: 입력 데이터의 차원을 변경하여 모델에 맞게 변환한다

model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),input_shape=[None]),
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
# Set the learning rate
learning_rate = 0.005

# Set the optimizer 
#optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, momentum=0.9)

# Set the training parameters
model.compile(loss=tf.keras.losses.Huber(),
              optimizer='adam',
              metrics=["mae"])

# Train the model
model.fit(X_train, y_train, epochs=100)
model.evaluate(X_test, y_test)
pred = model.predict(X_test)

# results
mse = mean_squared_error(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print('Tair MSE: {:.4f}'.format(mse)) 
print('Tair MAE: {:.4f}'.format(mae)) 
print('Tair RMSE: {:.4f}'.format(rmse)) 
print('Tair R2: {:.4f}'.format(r2))
