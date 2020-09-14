'''
Created on 13.09.2020

@author: Tobias Lang
'''

from translator.model_simple import ModelSimple
from translator.translator import Translator


def main():
    '''
    Calling training and Translation. No CLI yet.
    '''

    translator = Translator()

    # PATH, MAX_VOCAB, MAX_DEST_SENT, EMBED, EPOCHS
    # data_set = ("data/deu-eng/deu_1k_split.txt", 1000, 10, 100, 250)
    data_set = ("data/deu-eng/deu.txt", None, 10, 200, 2)
    model = ModelSimple()

    print("Running Translator Training...")
    translator.run_training(model, data_set)

    model_name = type(model).__name__
    model_path = "models/translator/{}/{}-{}-{}-{}-{}".format(model_name,
                                                              *data_set)
    dictionaries_path = "models/translator/{}/{}-{}-{}-{}-{}-dicts.pkl".format(model_name,
                                                                               *data_set)
    translator.run_translation(model_path, dictionaries_path)


if __name__ == "__main__":
    main()
