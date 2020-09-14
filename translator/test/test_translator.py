'''
Translator Unit-tests.

@author: Tobias Lang
'''

import os
import unittest

from tensorflow.keras.preprocessing.text import Tokenizer
from translator.translator import Translator


class TestTranslator(unittest.TestCase):
    '''
    As testing the running of the model would take to long,
    and be an application test, we just check the data-preparation
    pipeline.
    '''

    RESOURCE_BASE_PATH = ('../../data')
    FILE_PATH = os.path.join(RESOURCE_BASE_PATH, 'deu-eng/deu.txt')

    def test_prepare_data(self):
        '''
        Validate data preparation and preprocessing.
        '''

        prepared_data = Translator.prepare_data(self.FILE_PATH, 500, 6)

        # Unpack - Destination
        dest_data = prepared_data.get("destination_data", None)
        dest_tokenizer = prepared_data.get("destination_tokenizer", None)
        dest_vocab_size = prepared_data.get("destination_vocab_size", None)
        dest_max_length = prepared_data.get("destination_max_length", None)

        # Check
        self.assertIsNotNone(dest_data)
        self.assertEqual(dest_data.shape, (132599, 6, 1))
        self.assertIsInstance(dest_tokenizer, Tokenizer)
        self.assertEqual(dest_vocab_size, 500)
        self.assertEqual(dest_max_length, 6)

        # Unpack - Target
        targ_data = prepared_data.get("target_data", None)
        targ_tokenizer = prepared_data.get("target_tokenizer", None)
        targ_vocab_size = prepared_data.get("target_vocab_size", None)
        targ_max_length = prepared_data.get("target_max_length", None)
        # Check
        self.assertIsNotNone(targ_data)
        self.assertEqual(targ_data.shape, (132599, 15, 1))
        self.assertIsInstance(targ_tokenizer, Tokenizer)
        self.assertEqual(targ_vocab_size, 500)
        self.assertEqual(targ_max_length, 15)

    def test_prepare_data_max_vocab(self):
        '''
        Validate data preparation and preprocessing.
        '''

        prepared_data = Translator.prepare_data(self.FILE_PATH, None, 6)

        # Unpack - Destination
        dest_vocab_size = prepared_data.get("destination_vocab_size", None)
        dest_max_length = prepared_data.get("destination_max_length", None)

        # Check
        self.assertEqual(dest_vocab_size, 12229)
        self.assertEqual(dest_max_length, 6)

        # Unpack - Target
        targ_vocab_size = prepared_data.get("target_vocab_size", None)
        targ_max_length = prepared_data.get("target_max_length", None)
        # Check
        self.assertEqual(targ_vocab_size, 22350)
        self.assertEqual(targ_max_length, 19)

    def test_prepare_data_max_vocab_max_length(self):
        '''
        Validate data preparation and preprocessing.
        '''

        prepared_data = Translator.prepare_data(self.FILE_PATH, None, None)

        # Unpack - Destination
        dest_vocab_size = prepared_data.get("destination_vocab_size", None)
        dest_max_length = prepared_data.get("destination_max_length", None)

        # Check
        self.assertEqual(dest_vocab_size, 16668)
        self.assertEqual(dest_max_length, 101)

        # Unpack - Target
        targ_vocab_size = prepared_data.get("target_vocab_size", None)
        targ_max_length = prepared_data.get("target_max_length", None)
        # Check
        self.assertEqual(targ_vocab_size, 35317)
        self.assertEqual(targ_max_length, 76)


if __name__ == '__main__':
    unittest.main()
