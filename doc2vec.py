import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

class Doc2Vec():
    """
    3 layers NN
        - Input layer with n nodes - n == number of documents, made with one-hot
          encoding for the document number
        - Hidden layer with size p - p == size of paragraph vectors outputted
        - Activation function is weighted sum of input layer activations
        - Output layer with M  - M == number of unique words in vocab

    Run a softmax to get a classification 

    Paragraph vector refers to the p x n matrix with weights
    """

    def __init__():
        pass

    def train(data):
        pass

    def predict():
        pass