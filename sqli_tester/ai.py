import tensorflow as tf
import numpy as np
import random

import numpy as np
import tensorflow as tf
import random
payload_list = open('./sqli_tester/all.txt', 'r').readlines()
payload_to_int = {payload: i for i, payload in enumerate(payload_list)}

def int_to_payload(integer):
    """Convert an integer to a payload string. If the integer is not a valid label, return a default string."""
    try:
        return payload_list[integer]
    except IndexError:
        return payload_list[0]  # use default payload
class ModelClass:
    """ used for getting all the juicy payloads and making them """

    def __init__(self, model=None, model_file=None):
        if not model:
            self.model = tf.keras.Sequential([
                tf.keras.layers.LSTM(8, input_shape=(1, 1)),
                tf.keras.layers.Dense(2122, activation='relu'),
            ])
        else:
            self.model = model
        if model_file:
            self.model.load_weights(model_file)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
    def train(self):
            # Read the payloads from the file
        payload_list = open('./sqli_tester/all.txt', 'r').readlines()

    # Generate random seeds and labels
        seeds = np.random.rand(128).reshape(128, 1)
        labels = [random.choice(payload_list) for _ in range(128)]

    # Convert labels to integer values
        label_values = [payload_list.index(label) for label in labels]

    # Create a dataset from the seeds and labels
        dataset = tf.data.Dataset.from_tensor_slices((seeds, label_values))

    # Split the dataset into a training set and a validation set
        train_dataset = dataset.take(80)
        val_dataset = dataset.skip(80)

    # Convert the training and validation datasets to numpy arrays
        # Convert the training and validation datasets to numpy arrays
        train_data = np.array([x.numpy() for x, y in train_dataset])
        train_labels = np.array([y.numpy() for x, y in train_dataset])
        val_data = np.array([x.numpy() for x, y in val_dataset])
        val_labels = np.array([y.numpy() for x, y in val_dataset])
        num_classes = len(payload_list)
        train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
        val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=num_classes)

# Reshape the data to fit the input shape of the LSTM layer
        train_data = train_data.reshape(80, 1, 1)
        val_data = val_data.reshape(48, 1, 1)


    # Use the loss function and the optimizer to compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
        history = self.model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))


    # Save the model
        self.model.save('model.h5')
    
        return self.model
    def gen_payload(self):
        """ Generates a  payload by using a random seed """
        ran = random.randint(1, 128)
        pre = self.model.predict(np.array([[ran]]))
        payload = payload_list[np.argmax(pre)]
        return payload
if __name__ == '__main__':
    modelc = ModelClass(model_file='model.h5')
    modelc.train()
    model = modelc.model
    pay = modelc.gen_payload()
    print(pay)
    
