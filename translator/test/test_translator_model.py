'''
Created on 13.09.2020

@author: Tobias Lang
'''
import unittest
import logging
import tensorflow as tf
import numpy as np

from translator.model_simple import ModelSimple


class TestModel(unittest.TestCase):
    '''
    Testing of model generation, and cooccurrence matrix generation.
    '''

    def setUp(self):
        np.random.seed(42)
        tf.random.set_seed(42)

    @unittest.skip("Deactivated, just a first test, whether nor not the model is actually working")
    def test_train_fit(self):
        '''
        Train the model on random data using the .fit() method.
        Will differ from train_on_batch() due to random shuffeling of the batches.
        '''

        # Fix for Tensorflow/Keras GPU issue
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except ValueError:
            logging.warning("Invalid device or cannot modify virtual devices once initialized.")

        model = ModelSimple.create_model(input_vocab_size=81, embedding_dim=8,
                                         input_length=6, output_length=5,
                                         output_vocab_size=141, learning_rate=0.01)

        input_data = np.array([[[39], [0], [0], [0], [0], [0]],
                               [[39], [0], [0], [0], [0], [0]],
                               [[34], [0], [0], [0], [0], [0]],
                               [[34], [0], [0], [0], [0], [0]],
                               [[4], [0], [0], [0], [0], [0]],
                               [[78], [0], [0], [0], [0], [0]],
                               [[78], [0], [0], [0], [0], [0]]])

        output_data = np.array([[[31], [0], [0], [0], [0]],
                                [[31], [0], [0], [0], [0]],
                                [[34], [0], [0], [0], [0]],
                                [[34], [0], [0], [0], [0]],
                                [[73], [0], [0], [0], [0]],
                                [[19], [0], [0], [0], [0]],
                                [[140], [0], [0], [0], [0]]])

        loss = model.fit(input_data, output_data)
        print("Iteration {}, loss={}".format(1, loss.history['loss'][-1]))
        self.assertAlmostEqual(loss.history['loss'][-1], 4.949)

    @unittest.skip("Deactivated, time-consuming, and just a test to see the model summary.")
    @classmethod
    def test_model(cls):
        '''
        Check whether the model compiles or not.
        '''
        model = ModelSimple.create_model(input_vocab_size=80, embedding_dim=5,
                                         input_length=6, output_length=6,
                                         output_vocab_size=80, learning_rate=0.01)
        print(model.summary())


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
