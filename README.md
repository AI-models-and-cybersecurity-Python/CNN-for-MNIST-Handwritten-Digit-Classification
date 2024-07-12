# CNN for MNIST Handwritten Digit Classification

This project leverages TensorFlow and Keras libraries to construct and train a Convolutional Neural Network (CNN) model. The primary objective of the repository is to classify handwritten digits using the well-known MNIST dataset.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/AI-models-and-cybersecurity-Python/CNN-for-MNIST-Handwritten-Digit-Classification.git
    ```
2. Navigate to the project directory:
    ```bash
    cd CNN-for-MNIST-Handwritten-Digit-Classification
    ```
3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place the `mnist.npz` file in the project directory.
2. Run the script:
    ```bash
    python HandwritingRecognition.py
    ```

## Dataset

The dataset used for training the model should be in the `.npz` format and contain the following arrays:
- `x_train`
- `x_test`
- `y_train`
- `y_test`

## License

This project is licensed under the MIT License.
