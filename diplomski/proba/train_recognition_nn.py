import tensorflow as tf
import pandas as pd
import numpy as np
import random
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense

source_model = load_model('mp_hand_gesture')
model = Sequential()
for layer in source_model.layers[:-1]: # go through until last layer
    model.add(layer)
model.add(Dense(5, activation='softmax'))

# print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy')

def train_test_split(x, y, split=0.8):
    indices = np.array(range(len(x)))
    
    train_size = round(split * len(y))

    random.seed(12345)
    random.shuffle(indices)

    train_indices = indices[0:train_size]
    test_indices = indices[train_size:len(x)]

    x_train = x[train_indices, :]
    x_test = x[test_indices, :]
    y_train = y[train_indices, :]
    y_test = y[test_indices, :]
    
    return x_train, y_train, x_test, y_test

def drop_columns(dataframe, columns):
    for column in columns:
        try:
            dataframe = dataframe.drop(column, axis=1)
        except Exception as e:
            pass
    return dataframe

df = pd.read_csv("./data/german_sign_language.csv")

df = df[df["label"] < "f"]
col_idx = 2
while col_idx <= 62:
    df = df.drop("coordinate {}".format(col_idx), axis=1)
    col_idx += 3
# print(df.to_string())

y_df = pd.get_dummies(df["label"])
x_df = df.drop("label", axis=1)
# print(x_df)
# print(y_df)

x_train, y_train, x_val, y_val = train_test_split(x_df.to_numpy(), y_df.to_numpy())


model.fit(x_train, y_train, batch_size=32, validation_data = (x_val, y_val), epochs=200, verbose=1)
model.save("./my_hand_gesture")
