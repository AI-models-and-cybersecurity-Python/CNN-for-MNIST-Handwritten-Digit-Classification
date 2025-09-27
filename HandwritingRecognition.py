import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten

def load_and_visualize_data(filepath):
    mnist = np.load(filepath)
    train_x = mnist["x_train"]
    train_y = mnist["y_train"]
    plt.figure()
    plt.imshow(train_x[0], cmap='gray')
    plt.title(f"Label: {train_y[0]}")
    plt.show()

def preprocess_data(filepath):
    mnist = np.load(filepath)
    train_x = mnist["x_train"]
    test_x = mnist["x_test"]
    train_y = mnist["y_train"]
    test_y = mnist["y_test"]
    x_train = train_x.astype("float32") / 255.0
    x_test = test_x.astype("float32") / 255.0
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    y_train = to_categorical(train_y, 10)
    y_test = to_categorical(test_y, 10)
    return x_train, x_test, y_train, y_test

def build_model():
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, epochs=10, batch_size=128):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    return history

def evaluate_and_visualize(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.2f}")
    preds = model.predict(x_test)
    preds = np.argmax(preds, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    plt.figure()
    plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {preds[0]}, True: {y_test_labels[0]}")
    plt.show()

if __name__ == "__main__":
    filepath = "mnist.npz"
    load_and_visualize_data(filepath)
    x_train, x_test, y_train, y_test = preprocess_data(filepath)
    model = build_model()
    history = train_model(model, x_train, y_train, epochs=10, batch_size=128)
    evaluate_and_visualize(model, x_test, y_test)
