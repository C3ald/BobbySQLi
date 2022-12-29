import tensorflow as tf
import numpy as np
import random

class ModelClass:
    """ used for getting all the juicy payloads and making them """

    def __init__(self, model=None):
        if not model:
            self.model = tf.keras.Sequential([tf.keras.layers.Embedding(input_dim=256, output_dim=128),
                                              tf.keras.layers.LSTM(128),
                                              tf.keras.layers.Dense(
                                                  1, activation='sigmoid')
                                              ])
            self.model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

        else:
            self.model = model
    def train(self):
            payload_list = open('./payloads/all.txt', 'r').readlines()
            seeds = np.random.randint(0,high=2**32-1, size=(500, 1), dtype=np.int64)
            labels = [random.choice(payload_list) for _ in range(500)]
            dataset = tf.data.Dataset.from_tensor_slices((seeds, labels))
            train_dataset = dataset.take(80)
            val_dataset = dataset.skip(80)
            print(train_dataset.as_numpy_iterator())
            #inputs = tf.Tensor(1,(), dtype=tf.int64)
            train_dataset=np.reshape(train_dataset, train_dataset.shape[0], 1, train_dataset.shape[1])
            history = self.model.fit(train_dataset, epochs=10, validation_data=val_dataset, validation_split=0.1)
            # self.model.save('model.h5')
            return self.model

if __name__ == '__main__':
    modelc = ModelClass()
    modelc.train()