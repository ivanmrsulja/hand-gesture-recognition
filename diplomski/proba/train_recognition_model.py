import tensorflow as tf
import pandas as pd
import numpy as np
import random
import pickle
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

def train_and_evaluate_nn(x_train, y_train, x_val, y_val, save_model=False, verbose=True):
    source_model = load_model('mp_hand_gesture')
    model = Sequential()
    for layer in source_model.layers[:-1]: # go through until last layer
        model.add(layer)
    model.add(Dense(12, activation='softmax'))

    # print(model.summary())
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    model.fit(x_train, y_train, batch_size=32, validation_data = (x_val, y_val), epochs=200, verbose=verbose)
    
    if save_model:
        model.save("./my_hand_gesture")

def train_test_split(x, y, train_split=0.8, validation_split=0.1, join_validation=False):
    indices = np.array(range(len(x)))
    
    train_size = round(train_split * len(y))
    validation_size = round(validation_split * len(y))

    random.seed(12345)
    random.shuffle(indices)

    validation_indice_bound = train_size + validation_size
    train_indices = indices[0:train_size]
    validation_indices = indices[train_size:validation_indice_bound]
    test_indices = indices[validation_indice_bound:len(x)]

    x_train = x[train_indices, :]
    x_val = x[validation_indices, :]
    x_test = x[test_indices, :]
    y_train = y[train_indices, :]
    y_val = y[validation_indices, :]
    y_test = y[test_indices, :]

    if join_validation:
        x_train = np.concatenate((x_train, x_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def drop_columns(dataframe, columns):
    for column in columns:
        try:
            dataframe = dataframe.drop(column, axis=1)
        except Exception as e:
            pass
    return dataframe

def load_and_preprocess_data():
    df = pd.read_csv("./data/german_sign_language.csv")

    df = df[df["label"] < "n"]
    col_idx = 2
    while col_idx <= 62:
        df = df.drop("coordinate {}".format(col_idx), axis=1)
        col_idx += 3
    # print(df.to_string())

    y_df = pd.get_dummies(df["label"])
    x_df = df.drop("label", axis=1)
    # print(x_df)
    # print(y_df)

    return x_df.to_numpy(), y_df.to_numpy()

def get_train_and_test_data(join_validation):
    data_x, data_y = load_and_preprocess_data()
    return train_test_split(data_x, data_y, join_validation=join_validation)


def train_and_evaluate_svm(x_train, y_train, x_val, y_val, verbose=True, save_model=False):
    y_1d = tf.argmax(y_train, axis=1).numpy()

    clf = SVC(kernel="rbf", verbose=verbose)
    clf.fit(x_train, y_1d)
    predictions = clf.predict(x_val)

    print(classification_report(tf.argmax(y_val, axis=1).numpy(), predictions))
    print(f1_score(tf.argmax(y_val, axis=1).numpy(), predictions, average="micro"))

    if save_model:
        with open("svm.pickle", "wb") as file:
            pickle.dump(clf, file)

def train_and_evaluate_random_forest(x_train, y_train, x_val, y_val, verbose=True, save_model=False):
    y_1d = tf.argmax(y_train, axis=1).numpy()

    clf = RandomForestClassifier(n_estimators=500, max_depth=100, verbose=verbose, n_jobs=-1, random_state=1234)
    clf.fit(x_train, y_1d)
    predictions = clf.predict(x_val)

    print(classification_report(tf.argmax(y_val, axis=1).numpy(), predictions))
    print(f1_score(tf.argmax(y_val, axis=1).numpy(), predictions, average="micro"))

    if save_model:
        with open("random_forest.pickle", "wb") as file:
            pickle.dump(clf, file)


if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, y_test = get_train_and_test_data(join_validation=False)
    # train_and_evaluate_nn(x_train, y_train, x_val, y_val, save_model=True)
    # train_and_evaluate_svm(x_train, y_train, x_test, y_test, save_model=True)
    train_and_evaluate_random_forest(x_train, y_train, x_test, y_test, save_model=True, verbose=False)
