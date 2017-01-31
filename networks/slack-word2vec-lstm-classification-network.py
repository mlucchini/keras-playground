import json
import numpy as np
import os
import re
import sys
from gensim.models.word2vec import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from multiprocessing import cpu_count
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from sklearn.model_selection import train_test_split

# nltk.download()

export_directory = 'data/slack-export'
authors_file_name = 'data/slack+authors.json'
word2vec_file_name = 'data/slack-export-word2vec'
model_weight_file_name = 'data/slack-word2vec-lstm-classification-model-weights.h5'
num_features = 500
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer('[\'a-zA-Z]+')
lemmatizer = WordNetLemmatizer()
message_max_num_words = 15

def command_line_check():
    if not os.path.exists(export_directory) or os.listdir(export_directory) == []:
        print('First, extract the channels history exported from Slack into %s' % export_directory)
        sys.exit(-1)

def tokenize(message):
    words = []
    for sentence in sent_tokenize(message):
        tokens = [lemmatizer.lemmatize(t.lower()) for t in tokenizer.tokenize(sentence) if t.lower() not in stop_words]
        words += tokens
    return words

def to_author_vector(author_id, authors):
    vector = np.zeros(len(authors)).astype('float32')
    idx = authors[author_id]['pos'] - 1
    vector[idx] = 1.0
    return vector

def get_or_create_w2v_model(sentences):
    if os.path.exists(word2vec_file_name):
        print('Loading existing word2vec...')
        w2v_model = Word2Vec.load(word2vec_file_name)
    else:
        print('Creating word2vec model...')
        w2v_model = Word2Vec(sentences, size=num_features, min_count=1, window=10, workers=cpu_count())
        w2v_model.init_sims(replace=True)
        w2v_model.save(word2vec_file_name)
    return w2v_model

def load_authors():
    print('Loading authors...')
    authors = {}
    collection = open(authors_file_name, 'r').read()
    for author in json.loads(collection):
        authors[author['id']] = author
    return authors

def ignore_message(message, authors):
    if 'user' not in message or 'text' not in message:
        return True
    if 'subtype' in message:
        return True
    if message['user'] not in authors:
        return True

def load_conversations(authors):
    print('Loading conversations...')
    messages_X = []
    messages_Y = []
    ignored_messages = 0
    valid_paths = re.compile('[0-9]{4}-[0-9]{2}-[0-9]{2}.json')
    paths = [os.path.join(root, f) for root, dirs, files in os.walk(export_directory) for f in files if valid_paths.match(f)]
    for path in paths:
        conversation = open(path, 'r').read()
        for message in json.loads(conversation):
            if not ignore_message(message, authors):
                messages_X.append(tokenize(message['text']))
                messages_Y.append(to_author_vector(message['user'], authors))
            else:
                ignored_messages += 1
    print('Loaded %d messages from the conversations and ignored %d' % (len(messages_X), ignored_messages))
    return messages_X, messages_Y

def convert_messages_to_w2v(messages_X, messages_Y, model):
    print('Converting messages to word2vec...')
    X = np.zeros(shape=(len(messages_X), message_max_num_words, num_features)).astype('float32')
    Y = np.zeros(shape=(len(messages_Y), len(authors))).astype('float32')
    empty_word = np.zeros(num_features).astype('float32')
    for idx, message in enumerate(messages_X):
        for jdx, word in enumerate(message):
            if jdx == message_max_num_words:
                break
            else:
                if word in model:
                    X[idx, jdx, :] = model[word]
                else:
                    X[idx, jdx, :] = empty_word
    for idx, author_vector in enumerate(messages_Y):
        Y[idx, :] = author_vector
    return X, Y

def split_data(x, y):
    return train_test_split(x, y, test_size=0.3)

def generate_model(x, y, authors):
    print('Creating model...')
    model = Sequential()
    model.add(LSTM(int(message_max_num_words * 1.5), input_shape=(message_max_num_words, num_features)))
    model.add(Dropout(0.3))
    model.add(Dense(len(authors), activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, x, y, x_test, y_test):
    print('Training model...')
    model.fit(x, y, batch_size=128, nb_epoch=5, validation_data=(X_test, Y_test))

def evaluate_model(model, x_test, y_test):
    print('Evaluating model...')
    score, acc = model.evaluate(x_test, y_test, batch_size=128)
    print('Score: %1.4f' % score)
    print('Accuracy: %1.4f' % acc)

def predict_author_from_user_input(model, w2v_model, authors):
    while True:
        text = raw_input('Please enter a quote: ')
        messages_X = [tokenize(text)]
        X, Y = convert_messages_to_w2v(messages_X, [], w2v_model)
        preds = model.predict(X)[0]
        for idx, score in enumerate(preds):
            id = next(x for x in authors if authors[x]['pos'] == idx + 1)
            print("%2.2f for %s" % (score * 100, authors[id]['name']))

def save_model(model):
    print('Saving model...')
    model.save_weights(model_weight_file_name)

def load_model(model):
    print('Loading model...')
    model.load_weights(model_weight_file_name)

command_line_check()
authors = load_authors()
messages_X, messages_Y = load_conversations(authors)
w2v_model = get_or_create_w2v_model(messages_X)
X, Y = convert_messages_to_w2v(messages_X, messages_Y, w2v_model)
X_train, X_test, Y_train, Y_test = split_data(X, Y)
model = generate_model(X_train, Y_train, authors)

if not os.path.exists(model_weight_file_name):
    train_model(model, X_train, Y_train, X_test, Y_test)
    evaluate_model(model, X_test, Y_test)
    save_model(model)
else:
    load_model(model)
    predict_author_from_user_input(model, w2v_model, authors)
