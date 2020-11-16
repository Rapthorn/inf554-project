
from collections import defaultdict
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow
import tensorflow_hub as hub

from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error

import tensorflow_hub as hub
from verstack.stratified_continuous_split import scsplit

# embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)


embed = tf.saved_model.load("./model/")
train_data = pd.read_csv("./data/train.csv")

X_train, X_test, y_train, y_test = scsplit(train_data, train_data['retweet_count'], stratify=train_data['retweet_count'], train_size=0.75, test_size=0.25)
X_train = X_train.drop(['retweet_count'], axis=1)
X_test = X_test.drop(['retweet_count'], axis=1)

X_train = X_train["text"]
X_test = X_test["text"]
print(X_train[:10])
print(embed(X_train[:10]))

def build_model():
    input_text = tf.keras.Input(shape=(1,), dtype="string")
    embedding = tf.keras.layers.Lambda(lambda x: embed(x)[0], output_shape=(512, ) )(input_text)
    dense = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(embedding)
    pred = tf.keras.layers.Dense(1, activation='relu')(dense)
    model = tf.keras.Model(inputs=[input_text], outputs=pred)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError])
    return model

model_elmo = build_model()
print(model_elmo)


if __name__ == "__main__":
    # print(embeddings.shape)
    print()
