"""
Doc2Vec model in Keras. Started in tensorflow with the Graph interface, but
switched back for ease of use
"""

import tensorflow as tf
import numpy as np
from keras.layers import Input, Embedding, Dense, Flatten, Concatenate
from keras.models import Functional
from sklearn.model_selection import train_test_split

class Doc2Vec:
    """
    3 layers NN
        - Input layer with n nodes - n == number of documents, made with one-hot
          encoding for the document number
        - Hidden layer with size p - p == size of paragraph vectors outputted
        - Activation function is weighted sum of input layer activations
        - Output layer with M  - M == number of unique words in vocab
    """

    def __init__(self, docs, embed_size = 16, window_size = 3):
        # Save various fields for operations and set up docs for later
        self.docs = docs
        self.embed_size = embed_size
        self.window_size = window_size
        self.prepare_docs()

        self.doc_ids = len(self.doc_vocab)
        self.word_vocab = len(self.text_vocab)


        # Build model - Start with the embeddings combined
        docID = Input(3,)
        window = Input(shape=(self.window_size,))
        doc_embed = Embedding(input_dim=self.doc_ids, output_dim=self.embed_size, input_length=1)(docID)
        window_embed = Embedding(input_dim=self.word_vocab, output_dim=self.embed_size, input_length=self.window_size)(window)

        # Flattened layer contains the concatenated window_embed and doc_embed
        flat = Flatten()(Concatenate()([doc_embed, window_embed]))
        hidden = Dense(self.embed_size, activation="relu") (flat)
        output = Dense(1, activation="sigmoid") (hidden)
        model = Functional(inputs= [docID, window], outputs=output)
        model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics = "accuracy")
        self.model = model

    def prepare_docs(self):
        # Generate sliding windows for each document
        for doc in self.docs:
            doc.make_window(self.window_size)

        # Make vocab sets for ids and overall text
        doc_vocab = []
        text_vocab = set([""])
        for doc in self.docs:
            doc_vocab.append(doc.doc_id)
            text_vocab.update(doc.text)

        self.doc_vocab = doc_vocab
        self.text_vocab = text_vocab

        # Label items
        self.doc_label = self.label_item(self.doc_vocab)
        self.text_label = self.label_item(self.text_vocab)
        self.indices = list(range(len(self.docs)))

    def label_item(self, items):
        # Encode words/ids corresponding to a number
        encoding = {}
        for i, item in enumerate(items):
            encoding[item] = i
        return encoding

    def train(self, epochs, learning_rate):
        data = self.get_batch_data()
        x_train, x_test, y_train, y_test = train_test_split(data[0][0], data[1], test_size = 0.33, random_state = 22)
        x_train=np.asarray(x_train[0])
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=10)
        # self.model.fit_generator(generator = self.get_batch_data, epochs= epochs)

    def predict(self):
        pass

    def get_batch_data(self):
        docs = []
        words = []
        ids = []
        for doc in self.docs:
            # Pull the paragraph data needed for each doc
            temp_doc = []
            temp_word = []
            temp_out = []
            encoding = {}
            for item, label in self.doc_label.items():
                encoding.update({item: label * doc.doc_id})
            for sliding_window in doc.windows:
                temp_word_2 = []
                for word in sliding_window:
                    next = {}
                    for item, label in self.text_label.items():
                        next.update({item: label * word})
                    temp_word_2.append(next)
                temp_doc.append(encoding)
                temp_word.append(temp_word_2)
                temp_out.append(1)
            docs.append(temp_doc)
            words.append(temp_word)
            ids.append(temp_out)
        
        return [np.array(docs), np.array(words)], np.array(ids)