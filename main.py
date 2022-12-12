from doc2vec import Doc2Vec
from doc import Doc
import csv

EPOCHS = 200
LEARNING_RATE = 0.001

corpus = []

with open("movie_plots.csv") as file:
    reader = csv.reader(file)
    entries = 100
    for row in reader:
        if entries <= 0:
            break
        corpus.append(row[2])
        entries -= 1

# Remove punctuation, numbers, and stop words

stop_words = ["is", "a", "the", "will", "be"]
final_words = []
for text in corpus:
    temp_split = text.split(" ")
    for stop_word in stop_words:
        if stop_word in temp_split:
            temp_split.remove(stop_word)
    final_words.append(temp_split)

# Data fully cleaned, can send into the Docs and onto the model
final_docs = []
doc_id = 0
for text in final_words:
    final_docs.append(Doc(doc_id, str(text)))
    doc_id += 1

model = Doc2Vec(final_docs)
model.train(EPOCHS, LEARNING_RATE)