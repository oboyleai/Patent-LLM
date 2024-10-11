import random
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from IPython.display import Image
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional
from keras.models import Sequential, load_model
import gc
import sklearn
import pickle
from sklearn.utils import shuffle
import re
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
# from google.colab import drive


data = pd.read_csv('./neural_network_patent_query.csv',
                   parse_dates=['patent_date'])
original_abstracts = list(data['patent_abstract'])


def format_patent(patent):
    """Add spaces around punctuation
    and remove references to images/citations."""

    # Add spaces around punctuation
    patent = re.sub(r'(?<=[^\s0-9])(?=[.,;?])', r' ', patent)

    # Remove references to figures
    patent = re.sub(r'\((\d+)\)', r'', patent)

    # Remove double spaces
    patent = re.sub(r'\s\s', ' ', patent)
    return patent


# formatted = []

# # Iterate through all the original abstracts and apply formatting
# for a in original_abstracts:
#     formatted.append(format_patent(a))

# # Features and Labels
# # Note to self: new_texts and new_sequences are the texts and corresponding sequences
# # from only the abstracts that are 50 or more words (a sequence is vector of ints representing a piece of text)


# def make_sequences(texts,
#                    training_length=50,
#                    lower=True,
#                    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
#     """Turn a set of texts into sequences of integers"""

#     # Create the tokenizer object and train on texts
#     tokenizer = Tokenizer(lower=lower, filters=filters)
#     tokenizer.fit_on_texts(texts)

#     # Create look-up dictionaries and reverse look-ups
#     word_idx = tokenizer.word_index
#     idx_word = tokenizer.index_word
#     num_words = len(word_idx) + 1
#     word_counts = tokenizer.word_counts

#     print(f'There are {num_words} unique words.')

#     # Convert text to sequences of integers
#     sequences = tokenizer.texts_to_sequences(texts)

#     # Limit to sequences with more than training length tokens
#     seq_lengths = [len(x) for x in sequences]
#     over_idx = [
#         i for i, l in enumerate(seq_lengths) if l > (training_length + 20)
#     ]

#     new_texts = []
#     new_sequences = []

#     # Only keep sequences with more than training length tokens
#     for i in over_idx:
#         new_texts.append(texts[i])
#         new_sequences.append(sequences[i])

#     training_seq = []
#     labels = []

#     # Iterate through the sequences of tokens
#     for seq in new_sequences:

#         # Create multiple training examples from each sequence
#         for i in range(training_length, len(seq)):
#             # Extract the features and label
#             extract = seq[i - training_length:i + 1]

#             # Set the features and label
#             training_seq.append(extract[:-1])
#             labels.append(extract[-1])

#     print(f'There are {len(training_seq)} training sequences.')

#     # Return everything needed for setting up the model
#     return word_idx, idx_word, num_words, word_counts, new_texts, new_sequences, training_seq, labels


# TRAINING_LENGTH = 50
# filters = '!"%;[\\]^_`{|}~\t\n'
# word_idx, idx_word, num_words, word_counts, abstracts, sequences, features, labels = make_sequences(
#     formatted, TRAINING_LENGTH, lower=False, filters=filters)

# # Create training and validation sets. One-hot encoding labels


# def create_train_valid(features,
#                        labels,
#                        num_words,
#                        train_fraction=0.7):
#     """Create training and validation features and labels."""

#     # Randomly shuffle features and labels
#     features, labels = shuffle(features, labels, random_state=50)

#     # Decide on number of samples for training
#     train_end = int(train_fraction * len(labels))

#     train_features = np.array(features[:train_end])
#     valid_features = np.array(features[train_end:])

#     train_labels = labels[:train_end]
#     valid_labels = labels[train_end:]

#     # Convert to arrays
#     X_train, X_valid = np.array(train_features), np.array(valid_features)

#     # Using int8 for memory savings
#     y_train = np.zeros((len(train_labels), num_words), dtype=np.int8)
#     y_valid = np.zeros((len(valid_labels), num_words), dtype=np.int8)

#     # One hot encoding of labels
#     for example_index, word_index in enumerate(train_labels):
#         y_train[example_index, word_index] = 1

#     for example_index, word_index in enumerate(valid_labels):
#         y_valid[example_index, word_index] = 1

#     # Memory management: one-hot encoding the labels creates massive numpy arrays
#     # so I took care to delete the un-used objects from the workspace.
#     import gc
#     gc.enable()
#     del features, labels, train_features, valid_features, train_labels, valid_labels
#     gc.collect()

#     return X_train, X_valid, y_train, y_valid


# X_train, X_valid, y_train, y_valid = create_train_valid(
#     features, labels, num_words)


# # Save the intermediate results as it is consuming too much memory!!


# def save_intermediate_results(datap):
#     for i in data:
#         with open(f'./Datasets/{i}.pkl', 'wb') as f:
#             pickle.dump(globals()[i], f)


# data = ['word_idx', 'idx_word', 'num_words', 'word_counts',
#         'abstracts', 'sequences', 'features', 'labels', 'X_valid', 'y_valid']
# save_intermediate_results(data)

# with open('./Datasets/X_valid.pkl', 'rb') as f:
#     X_valid = pickle.load(f)
#     print(X_valid.shape)

# # Clean up
# gc.enable()
# del (word_idx, idx_word, num_words, word_counts, abstracts,
#      sequences, features, labels, X_valid, y_valid)
# gc.collect()


# # Build Model: After converting the words into embeddings, we pass them through
# # a single LSTM layer, then into a fully connected layer with relu activation
# # before the final output layer with a softmax activation


# def make_word_level_model(num_words,
#                           lstm_cells=64,
#                           trainable=True,
#                           lstm_layers=1,
#                           bi_direc=False):
#     """Make a word level recurrent neural network with option for pretrained embeddings
#        and varying numbers of LSTM cell layers."""

#     model = Sequential()

#     model.add(
#         Embedding(
#             input_dim=num_words,
#             output_dim=100,
#             input_length=50,
#             trainable=True))

#     # If want to add multiple LSTM layers
#     if lstm_layers > 1:
#         for i in range(lstm_layers - 1):
#             model.add(
#                 LSTM(
#                     lstm_cells,
#                     return_sequences=True,
#                     dropout=0.1,
#                     recurrent_dropout=0.1))

#     # Add final LSTM cell layer
#     if bi_direc:
#         model.add(
#             Bidirectional(
#                 LSTM(
#                     lstm_cells,
#                     return_sequences=False,
#                     dropout=0.1,
#                     recurrent_dropout=0.1)))
#     else:
#         model.add(
#             LSTM(
#                 lstm_cells,
#                 return_sequences=False,
#                 dropout=0.1,
#                 recurrent_dropout=0.1))

#     model.add(Dense(128, activation='relu'))

#     # Dropout for regularization
#     model.add(Dropout(0.5))

#     # Output layer
#     model.add(Dense(num_words, activation='softmax'))

#     # Compile the model
#     model.compile(
#         optimizer='adam',
#         loss='categorical_crossentropy',
#         metrics=['accuracy'])

#     return model


# model = make_word_level_model(
#     num_words=16192,
#     lstm_cells=64,
#     trainable=True,
#     lstm_layers=1)
# model.summary()

# plot_model(model, to_file=f'./Datasets/patent abstract.png', show_shapes=True)

# # Train Model!
# # Early Stopping: Stop training when validation loss no longer decreases
# # Model Checkpoint: Save the best model on disk

# # Patience: Number of epochs with no improvement after which training will be stopped


# def make_callbacks(model_name, save=True):
#     """Make list of callbacks for training"""
#     # callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
#     callbacks = []
#     if save:
#         callbacks.append(
#             ModelCheckpoint(
#                 f'./Datasets/patent_abstracts.h5',
#                 save_best_only=True,
#                 save_weights_only=False))
#     return callbacks


# callbacks = make_callbacks(model)

# # Verbose is for showing output
# # Batch_size defines the number of samples that will be propagated through the network
# # Typically networks train faster with mini-batches. That's because we update the weights after each propagation.
# # If we used all samples during propagation we would make only 1 update for the network's parameter.

# with open('./Datasets/X_valid.pkl', 'rb') as f:
#     X_valid = pickle.load(f)
#     print(X_valid.shape)

# with open('./Datasets/y_valid.pkl', 'rb') as f:
#     y_valid = pickle.load(f)
#     print(y_valid.shape)

# history = model.fit(
#     X_train,
#     y_train,
#     epochs=5,
#     batch_size=128,
#     verbose=1,
#     callbacks=callbacks,
#     validation_data=(X_valid, y_valid))

# model.save('patents_abstracts')

def load_values(data):
    val = []
    for i in data:

        with open(f'./Datasets/{i}.pkl', 'rb') as f:
            globals()[f'{i}'] = pickle.load(f)
            print(type(globals()[f'{i}']))
            val.append(globals()[f'{i}'])
    return val


X_valid, y_valid = load_values(['X_valid', 'y_valid'])
X_valid.shape


model = load_model("patents_abstracts")
results = model.evaluate(X_valid, y_valid, batch_size=128)

with open('./Datasets/idx_word.pkl', 'rb') as f:
    idx_word = pickle.load(f)

with open('./Datasets/sequences.pkl', 'rb') as f:
    sequences = pickle.load(f)

print(len(sequences))
print(sequences[0])
str_out = ""
for i in sequences[0]:
    str_out += idx_word.get(i) + " "


def generate_output(model,
                    sequences,
                    training_length=75,
                    new_words=50,
                    diversity=1,
                    return_output=True,
                    n_gen=1):
    """Generate `new_words` words of output from a trained model and format into HTML."""

    # Choose a random sequence
    seq = random.choice(sequences)

    # Choose a random starting point
    seed_idx = random.randint(0, len(seq) - training_length - 10)
    # Ending index for seed
    end_idx = seed_idx + training_length

    gen_list = []

    for n in range(n_gen):
        # Extract the seed sequence
        seed = seq[seed_idx:end_idx]
        original_sequence = [idx_word[i] for i in seed]
        generated = seed[:] + ['#']

        # Find the actual entire sequence
        actual = generated[:] + seq[end_idx:end_idx + new_words]

        # Keep adding new words
        for i in range(new_words):

            # Make a prediction from the seed
            preds = model.predict(np.array(seed).reshape(1, -1))[0].astype(
                np.float64)

            # Diversify
            preds = np.log(preds) / diversity
            exp_preds = np.exp(preds)

            # Softmax
            preds = exp_preds / sum(exp_preds)

            # Choose the next word
            probas = np.random.multinomial(1, preds, 1)[0]

            next_idx = np.argmax(probas)

            # New seed adds on old word
            seed = seed[1:] + [next_idx]
            generated.append(next_idx)

        # Showing generated and actual abstract
        n = []

        for i in generated:
            n.append(idx_word.get(i, ''))

        gen_list.append(n)

    a = []

    for i in actual:
        a.append(idx_word.get(i, ''))

    a = a[training_length:]

    gen_list = [
        gen[training_length:training_length + len(a)] for gen in gen_list
    ]

    if return_output:
        return original_sequence, gen_list, a


original_sequence, predicted_continuation, actual_continuation = generate_output(
    model, sequences, 50)

# display stuff


def header(text, color='black'):
    raw_html = f'<h1 style="color: {color};"><center>' + \
        str(text) + '</center></h1>'
    return raw_html


def box(text):
    raw_html = '<div style="border:1px inset black;padding:1em;font-size: 20px;">' + \
        str(text)+'</div>'
    return raw_html


def addContent(old_html, raw_html):
    old_html += raw_html
    return old_html


def remove_weird_grammar(str):
    return str.replace(' , ', ', ').replace(' . ', '. ')


seed_seq_string = ' '.join(i for i in original_sequence)
seed_seq_string = remove_weird_grammar(seed_seq_string)
pred_seq_string = ' '.join(i for i in predicted_continuation[0])
pred_seq_string = remove_weird_grammar(pred_seq_string)
actual_seq_string = ' '.join(i for i in actual_continuation)
actual_seq_string = remove_weird_grammar(actual_seq_string)

print('Beginning:')
print(seed_seq_string)
print('Predicted:')
print(pred_seq_string)
print()
print()
print()
print('Beginning:')
print(seed_seq_string)
print('Actual:')
print(actual_seq_string)
