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
        self.assertEqual(dest_vocab_size, 499)
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
        self.assertEqual(targ_vocab_size, 499)
        self.assertEqual(targ_max_length, 15)


if __name__ == '__main__':
    unittest.main()
