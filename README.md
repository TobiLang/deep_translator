# deep_translator

Project to evaluate sequential machine learning translators.

## Data

I have taken the ENG-DEU translation file from "http://www.manythings.org/anki/":http://www.manythings.org/anki/
Check out the *_about.txt* file for more information.

## Preprocessor

## Translator

First step was to get a basic pipeline going and be able to run training/evaluation on the data.

Currently, I have no Train/Dev/Test sets. Before implementing the sampled model, I will extend the preprocessor to to so.
Otherwise, checking performance of the different model would be hard.

### Model_Simple
A simple bidirectional RNN with input embedding and short term memory via LSTM.

### Model_Sampled
TBD: Adjust the simple model by using SampledSoftmaxLos (tf.nn.sampled_softmax_loss()) during training.

### Model_Search
TBD: This will extend the simple model by using BeamSearch.

### Model_Attention
TBD: Again, extend the Model using Attenion