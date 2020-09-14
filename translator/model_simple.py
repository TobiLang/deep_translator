'''
Simple Bidirectional RNN model translate sentences using Keras.

@author: Tobias Lang
'''

import logging
from time import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam


class ModelSimple:
    '''
    Simple Bidirectional RNN model translate sentences using Keras.
    '''

    @classmethod
    def create_model(cls, input_vocab_size=1000, embedding_dim=50, input_length=10, \
                     output_vocab_size=1000, output_length=10, learning_rate=0.01) -> Sequential:
        '''
        Create the model - Bi RNN with Embeddings and LSTM.

        param: vocabulary_size: Size of the given vocabulary data has been compiled against.
        param: embedding_dim: Desired dimension of the embedding.
        '''

        # Build the model
        model = Sequential(name="Bi-RNN_w_LSTM")

        # Embedding
        model.add(Embedding(input_dim=input_vocab_size, output_dim=embedding_dim,
                            input_length=input_length))

        # Encoder
        model.add(Bidirectional(LSTM(256)))
        model.add(RepeatVector(output_length))

        # Decoder
        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(TimeDistributed(Dense(512, activation='relu')))
        model.add(Dropout(0.5))  # Regularize
        model.add(TimeDistributed(Dense(output_vocab_size, activation='softmax')))

        # Create and compile Model
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adam(learning_rate),
                      metrics=['accuracy'])

        return model

    @classmethod
    def train_model(cls, model, input_sequences, output_sequences, epochs=5):
        '''
        Train the model on a set of given input and output sentences.
        '''

        # Run Training
        start_time = time()
        loss = model.fit(input_sequences, output_sequences, batch_size=32,
                         verbose=1, validation_split=0.2, shuffle=True,
                         epochs=epochs)
        end_time = time()

        logging.info("Training finished: Iterations: %d, loss=%f, Run-Time: %d sec",
                     epochs, loss.history['loss'][-1], int(end_time - start_time))
