import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras
import matplotlib.pyplot as plt
from dataset import Dataset

if __name__ == "__main__":
    dataset = Dataset("/home/group18/augdataset/", 128)
    valset = Dataset("/home/group18/valdataset/", 128)
    print(dataset.numbatches())
    first_test_batch = int(dataset.numbatches() * 0.9)

    model = models.Sequential()
    model.add(layers.Conv2D(64, (5, 5), activation = "relu", padding = "same", input_shape = (64, 64, 3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (5, 5), activation = "relu", padding = "same"))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation = "relu"))
    model.add(layers.Dense(7, activation = "softmax")) # output
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

    model.summary()

    callbacks = \
    [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    ]

    history = model.fit(dataset, validation_data = valset, epochs = 100, callbacks = callbacks)
    plt.plot(history.history['accuracy'], label='accuracy')
    # plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    # test_loss, test_acc = model.evaluate(dataset, verbose=2)

    model.save("/home/group18/models/model2.h5")