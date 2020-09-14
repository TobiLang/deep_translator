'''
Check Normalization and preprocessing.

@author: Tobias Lang
'''
import unittest

import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from preprocessor.text_preprocessor import TextPreprocessor


class TestTextPreprocessing(unittest.TestCase):
    '''
    Check Normalization and PreProcessing.
    '''

    RESOURCE_BASE_PATH = ('../../data')
    FILE_PATH = os.path.join(RESOURCE_BASE_PATH, 'deu-eng/deu.txt')

    def test_read(self):
        '''
        Testing File reading.
        '''
        expected = ('Hallo!\tCC-BY 2.0 (France) Att')
        doc = TextPreprocessor.read_file(self.FILE_PATH)

        self.assertEqual(len(doc), 34382099)
        self.assertEqual(doc[91:120], expected)

    def test_preprocess_deu_eng(self):
        '''
        Testing Preprocessing on Deu-Eng pairs.
        '''
        # Set Input
        deu_eng = ('Go.\tGeh.\tCC-BY 2.0 (France) Attribution: tatoeba.org #2877272\n'
                   'Hi.\tHallo!\tCC-BY 2.0 (France) Attribution: tatoeba.org #538123\n'
                   'I\'m game.\tIch bin dabei.\tCC-BY 2.0 (France)\n'
                   'Please keep me updated.\tBitte halten Sie mich auf dem Laufenden.\tCC-BY\n')

        # Expected
        expected_destination = [['go'], ['hi'], ['im', 'game']]
        expected_target = [['geh'], ['hallo'], ['ich', 'bin', 'dabei']]

        destination, target = TextPreprocessor.preprocess(deu_eng, 3)

        self.assertEqual(destination, expected_destination)
        self.assertEqual(target, expected_target)

    def test_preprocess_deu_eng_none_length(self):
        '''
        Testing Preprocessing on Deu-Eng pairs.
        '''
        # Set Input
        deu_eng = ('Go.\tGeh.\tCC-BY 2.0 (France) Attribution: tatoeba.org #2877272\n'
                   'Hi.\tHallo!\tCC-BY 2.0 (France) Attribution: tatoeba.org #538123\n'
                   'I\'m game.\tIch bin dabei.\tCC-BY 2.0 (France)\n'
                   'Please keep me updated.\tBitte halten Sie mich auf dem Laufenden.\tCC-BY\n')

        # Expected
        expected_destination = [['go'], ['hi'], ['im', 'game'], ['please', 'keep', 'me', 'updated']]
        expected_target = [['geh'], ['hallo'], ['ich', 'bin', 'dabei'],
                           ['bitte', 'halten', 'sie', 'mich', 'auf', 'dem', 'laufenden']]

        # Do return all sentences lengths
        destination, target = TextPreprocessor.preprocess(deu_eng, None)

        self.assertEqual(destination, expected_destination)
        self.assertEqual(target, expected_target)

    def test_preprocess_deu_eng_from_file(self):
        '''
        Testing Preprocessing on Deu-Eng pairs from a file.
        '''
        # Set Zarathustra (not in NLTK corpus)
        input_doc = TextPreprocessor.read_file(self.FILE_PATH)

        # Expected
        expected_destination = [['go'], ['hi'], ['hi'], ['run']]
        expected_target = [['geh'], ['hallo'], ['grüß', 'gott'], ['lauf']]

        destinations, targets = TextPreprocessor.preprocess(input_doc)

        self.assertEqual(destinations[0:4], expected_destination)
        self.assertEqual(targets[0:4], expected_target)

    def test_convert_to_dictionary(self):
        '''
        Testing convertion of a normalized doc to a dictionary and mapped dataset.
        '''
        # Setup small norm_doc
        norm_doc = [['lie', 'low'],
                    ['lock', 'it'],
                    ['i', 'loved', 'that', 'house'],
                    ['may', 'i', 'speak', 'to', 'you', 'outside', 'for', 'a', 'minute'],
                    ["Hooray"]]

        expected_data = np.array([[2, 3],
                                  [4, 5],
                                  [1, 6, 7, 8],
                                  [9, 1, 10, 11, 12, 13, 14, 15, 16],
                                  [17]])

        result_tuple = TextPreprocessor.convert_to_dictionary(norm_doc, 100)
        data = result_tuple[0]
        tokenizer = result_tuple[1]
        vocab_size = result_tuple[2]
        max_length = result_tuple[3]

        self.assertTrue((data == expected_data).all())
        self.assertIsInstance(tokenizer, Tokenizer)
        self.assertLessEqual(len(tokenizer.word_index), 100)
        self.assertEqual(vocab_size, 18)
        self.assertEqual(max_length, 9)

    def test_map_to_vector(self):
        '''
        Map a given sentence to a padded input vector.
        '''

        norm_doc = [['lie', 'low'],
                    ['lock', 'it'],
                    ['i', 'loved', 'that', 'house'],
                    ['may', 'i', 'speak', 'to', 'you', 'outside', 'for', 'a', 'minute']]
        max_length = 9
        result_tuple = TextPreprocessor.convert_to_dictionary(norm_doc)
        tokenizer = result_tuple[1]

        input_sentence = "I loved you, outside for a minute."
        expected_vector = np.asarray([[[1], [6], [12], [13], [14], [15], [16], [0], [0]]])

        mapped_vector = TextPreprocessor.map_to_vector(input_sentence, tokenizer, max_length)
        self.assertTrue((mapped_vector == expected_vector).all())

    def test_map_to_string(self):
        '''
        Map a given vector to an unpadded string.
        '''

        norm_doc = [['lie', 'low'],
                    ['lock', 'it'],
                    ['i', 'loved', 'that', 'house'],
                    ['may', 'i', 'speak', 'to', 'you', 'outside', 'for', 'a', 'minute']]
        result_tuple = TextPreprocessor.convert_to_dictionary(norm_doc)
        tokenizer = result_tuple[1]

        input_vector = np.asarray([[1, 6, 12, 13, 14, 15, 16, 0, 0]])
        expected_sentence = ["i loved you outside for a minute"]

        mapped_sentence = TextPreprocessor.map_to_string(input_vector, tokenizer)
        self.assertEqual(mapped_sentence, expected_sentence)

    def test_padding(self):
        '''
        Testing convertion of a normalized doc to a dictionary and mapped dataset.
        '''
        # Setup small norm_doc
        sequences = np.array([[2, 3],
                              [4, 5],
                              [1, 6, 7, 8],
                              [9, 1, 10, 11, 12, 13, 14, 15, 16]])

        expected_sequences = np.array([[2, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [4, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 6, 7, 8, 0, 0, 0, 0, 0, 0],
                                       [9, 1, 10, 11, 12, 13, 14, 15, 16, 0]])

        padding_length = 10
        padded_sequences = TextPreprocessor.add_padding(sequences, padding_length)

        self.assertTrue((padded_sequences == expected_sequences).all())


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
