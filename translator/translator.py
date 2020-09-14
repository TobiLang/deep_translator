'''
Translator for sentence translation using:
  * ModelSimple: Bidirectional RNN with Embeddings and LSTM
  * ModelSearch: ModelSimple extended by BeamSearch

@author: Tobias Lang
'''

import logging
from typing import Dict
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from preprocessor.text_preprocessor import TextPreprocessor as text_pre


class Translator:
    '''
    Translate sentences using RNNs.
    '''

    def __init__(self):
        # Set Logging to info
        logging_format = '%(asctime)-15s %(message)s'
        logging.basicConfig(level=logging.INFO, format=logging_format)

        # Fix for Tensorflow/Keras GPU issue
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except ValueError:
            logging.warning("Invalid device or cannot modify virtual devices once initialized.")

    @classmethod
    def __save_model_params(cls, model: Sequential,
                            params: Dict,
                            dest_tokenizer: Tokenizer, target_tokenizer: Tokenizer):
        '''
        Save the model and necessary dictionaries to disk.
        '''

        # Saving model and params
        model_path = "models/translator/{}/{}-{}-{}-{}-{}".format(type(model).__name__,
                                                                  params["input_file"],
                                                                  params["max_vocab_size"],
                                                                  params["max_length"],
                                                                  params["embedding_dim"],
                                                                  params["epochs"])
        model_dictionaries_path = "{}-dicts.pkl".format(model_path)

        model.save(model_path)
        logging.info("Saved model to: %s", model_path)

        # Saving params and dictionaries
        model_dictionaries = {"params": params,
                              "input_tokenizer": dest_tokenizer,
                              "output_tokenizer": target_tokenizer}
        with open(model_dictionaries_path, 'wb') as file:
            pickle.dump(model_dictionaries, file, pickle.HIGHEST_PROTOCOL)
        logging.info("Saved dictionaries to: %s", model_dictionaries_path)

    @classmethod
    def __load_model_params(cls, model_path: str, dictionaries_path: str) -> (Sequential, Dict):
        '''
        Load the model and necessary dictionaries from disk.
        '''

        model = load_model(model_path)

        with open(dictionaries_path, 'rb') as file:
            model_dictionaries = pickle.load(file)

        return model, model_dictionaries

    @classmethod
    def prepare_data(cls, input_file, max_vocab_size=None, dest_max_length=None) -> Dict:
        '''
        Preprocess a given input file, and convert it to a mapped dictionary.

        input_file: File doucment, having this structure:
                   DESTINATION_SENTENCE [TAB] TARGET_SENTENCE [TAB] ADD_INFO [NEWLINE]
        max_vocab_size: maximal vocabulary size to use.
        dest_max_length: Max length for DESTINATION_SENTENCE
        '''
        # Load data from disk
        document = text_pre.read_file(input_file)

        # Preprocess and normalize
        norm_destination, norm_target = text_pre.preprocess(document, dest_max_length)

        # Convert destination and target to vectors
        dest_tuple = text_pre.convert_to_dictionary(norm_destination, max_vocab_size)
        targ_tuple = text_pre.convert_to_dictionary(norm_target, max_vocab_size)
        # Unpack
        dest_data, dest_tok, dest_vocab_size, dest_length = (dest_tuple)
        targ_data, targ_tok, targ_vocab_size, targ_length = (targ_tuple)

        # Pad both vectors
        dest_data = text_pre.add_padding(dest_data, dest_length)
        targ_data = text_pre.add_padding(targ_data, targ_length)

        # Reshape Input - Sequential models need a 3-dimensional input:
        #  TrainSize x PadLength x 1 (word-int)
        dest_data = dest_data.reshape((-1, dest_data.shape[-1], 1))
        # Sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
        targ_data = targ_data.reshape(*targ_data.shape, 1)

        result_dict = {}
        dest_tuple = (dest_data, dest_tok, dest_vocab_size, dest_length)
        targ_tuple = (targ_data, targ_tok, targ_vocab_size, targ_length)
        for key, value in {'destination': dest_tuple, 'target': targ_tuple}.items():
            result_dict[key + "_data"] = value[0]
            result_dict[key + "_tokenizer"] = value[1]
            result_dict[key + "_vocab_size"] = value[2]
            result_dict[key + "_max_length"] = value[3]

        return result_dict

    def run_training(self, translator_model, data_set):
        '''
        Run Training

        Current-Hyperparameters:
            * Vocabulary size
            * Max sentence length
            * Embedding dims
            * Epochs
        '''

        # Unpack
        (input_file, max_vocab_size, max_length, embedding_dim, epochs) = data_set

        # Prepare Data and Vocabulary
        logging.info("Preprocessing data ...")
        prepared_data = self.prepare_data(input_file, max_vocab_size, max_length)

        # Generate Model
        logging.info("Generating model ...")
        input_vocab_size = prepared_data["destination_vocab_size"]
        input_length = prepared_data["destination_max_length"]
        output_vocab_size = prepared_data["target_vocab_size"]
        output_length = prepared_data["target_max_length"]
        logging.info(" ...with Params: Input (Voc,Len): %d,%d - Output (Voc,Len): %d/%d ",
                     input_vocab_size, input_length, output_vocab_size, output_length)
        model = translator_model.create_model(input_vocab_size, embedding_dim, input_length,
                                              output_vocab_size, output_length)

        # Train Model
        logging.info("Training model for %d epochs ...", epochs)
        translator_model.train_model(model,
                                     prepared_data["destination_data"],
                                     prepared_data["target_data"], epochs=epochs)

        logging.info("Saving model ...")
        model_params = {"input_file": input_file,
                        "max_vocab_size": max_vocab_size,
                        "max_length": max_length,
                        "input_length": input_length,
                        "output_length": output_length,
                        "embedding_dim": embedding_dim,
                        "epochs": epochs}
        self.__save_model_params(model, model_params,
                                 prepared_data["destination_tokenizer"],
                                 prepared_data["target_tokenizer"])

    def run_translation(self, model_path, dictionaries_path):
        '''
        Use a saved model (plus Dictionaries) to translate from Destination -> Target:
        '''

        model, model_dicts = self.__load_model_params(model_path, dictionaries_path)

        input_length = model_dicts.get("params").get("input_length")
        input_tokenizer = model_dicts.get("input_tokenizer")
        output_tokenizer = model_dicts.get("output_tokenizer")

        # Run while loop to handle inputs
        input_sentence = ''
        while input_sentence != 'exit':
            # Ask for input
            input_sentence = input("Sentence to translate or enter 'exit': ")

            if input_sentence == 'exit':
                break

            # Convert - longer than max_length will return None
            processed_input = text_pre.map_to_vector(input_sentence, input_tokenizer, input_length)

            # Do not process inputs longer than allowed length
            if len(processed_input) > input_length:
                print("Sorry, input longer than: %d", input_length)
                break

            pred = model.predict(processed_input)
            # Pred is a list of predictions for each input
            # We find the vector by using argmax on axis 1
            for entry in pred:
                vector = [np.argmax(entry, axis=-1)]
                output_sentence = text_pre.map_to_string(vector, output_tokenizer)
                print(output_sentence)
