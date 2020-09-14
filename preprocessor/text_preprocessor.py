'''
Simple Text preprocessor to prepare inputs for a Sequence translation model.

  Current shortcomings:
     * Does not handle apostrophes: I'm -> im.
     * Neither embeddings nor 1-Hot encoding are used.
     * Removes punctuation.

@author: Tobias Lang
'''

import sys
from typing import List
import string
import logging
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextPreprocessor:
    '''
    Text Preprocessing handles these steps:

      * Normalization:
        * Convert to lowercase
        * Remove punctuation.
        * Tokenize per word.

      * Preprocessing:
        * Split into destination and target languages (files are tab delimited).
        * Normalize.

      * Dictionary
        * Build dictionary for a given set of sentences.
        * Convert to dictionary representation.
        * Pad sentences to maximal length (given by destinations and targets).
    '''

    # Defing a padding length
    # PADDING_LENGTH = 15  # 30
    # Special Tokens
    UNK = '<unk>'
    UNK_IDX = 0
    PAD = '<pad>'
    PAD_IDX = 1

    @classmethod
    def read_file(cls, file_path: str) -> str:
        '''
        Read a text document from file.
        '''

        doc = None
        try:
            file_object = open(file_path, "r")
            doc = file_object.read()
            file_object.close()
        except OSError:
            logging.error("Could not open/read file: %s", file_path)
            sys.exit()

        return doc

    @classmethod
    def __normalize(cls, input_doc: str) -> List:
        '''
        Normalize given input_doc string.
        '''

        # To Lowercase and Strip Whitespaces
        stripped_doc = input_doc.lower().strip()
        # Remove punctuation
        cleaned_doc = stripped_doc.translate(str.maketrans('', '', string.punctuation))
        # cleaned_doc = stripped_doc.translate(None, string.punctuation)

        # Tokenize
        tokenizer = WordPunctTokenizer()
        tokens = tokenizer.tokenize(cleaned_doc)

        return tokens

    @classmethod
    def convert_to_dictionary(cls, normalized_doc: List, \
                              max_vocab_size=None) -> (List, Tokenizer, int, int):
        '''
        Given an normalized_doc (list of sentences, split into words), train a Tokenizer
        and turn given document into vectors based on the tokenizer.

        normalized_doc: List of normalized sentences to tokenize on.
        max_vocab_size: maximal vocabulary size to use.
        '''

        # Create tokenizer and fit on given texts
        tokenizer = Tokenizer(num_words=max_vocab_size, lower=True, char_level=False)
        tokenizer.fit_on_texts(normalized_doc)

        sequences = tokenizer.texts_to_sequences(normalized_doc)
        # Find actual vocab size and longest sequence
        vocab_size = np.max(np.max(sequences)) + 1  # Padding
        max_length = len(max(sequences, key=len))

        return sequences, tokenizer, vocab_size, max_length

    @classmethod
    def map_to_vector(cls, input_sentence: str, tokenizer: Tokenizer, max_length: int) -> List:
        '''
        Map a given sentence to a padded input vector.
        '''

        sentence = cls.__normalize(input_sentence)
        mapped_vector = tokenizer.texts_to_sequences([sentence])
        # Zero pad sentence up to max_sentence length
        padded_vector = cls.add_padding(mapped_vector, max_length)
        padded_vector = padded_vector.reshape((-1, padded_vector.shape[-1], 1))
        return padded_vector

    @classmethod
    def map_to_string(cls, input_vector: List, tokenizer: Tokenizer) -> List:
        '''
        Map a given vector to an unpadded string.
        '''

        return tokenizer.sequences_to_texts(input_vector)

    @classmethod
    def add_padding(cls, sequences, padding_length):
        '''
        Pad the given sequences.
        '''
        return pad_sequences(sequences, padding_length, padding='post')

    @classmethod
    def preprocess(cls, input_doc: List, destination_max_length=None) -> (List, List):
        '''
        Preprocess a given input.

        input_doc: Input document consisting of this structure:
                   DESTINATION_SENTENCE [TAB] TARGET_SENTENCE [TAB] ADD_INFO [NEWLINE]
        destination_max_length: Max length for DESTINATION_SENTENCE
        '''

        # Split input per line
        # DESTINATION_SENTENCE [TAB] TARGET_SENTENCE [TAB] ADD_INFO [NEWLINE]
        norm_dest = list()
        norm_targ = list()
        for entry in input_doc.splitlines():
            tabs = entry.split('\t')
            if len(tabs) > 2:
                # Normalize
                dest = cls.__normalize(tabs[0])
                targ = cls.__normalize(tabs[1])
                if not destination_max_length or len(dest) <= destination_max_length:
                    norm_dest.append(dest)
                    norm_targ.append(targ)

        return norm_dest, norm_targ
