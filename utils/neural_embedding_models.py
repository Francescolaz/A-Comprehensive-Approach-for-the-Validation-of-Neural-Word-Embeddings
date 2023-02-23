# Training neural embedding models


import os
import timeit
from gensim.test.utils import datapath
from gensim.models import Word2Vec, FastText
import datetime
from datetime import datetime
from traceback import format_exc


class NeuralEmbeddingModels():


    """
    A class used to train a neural embedding model


    Attributes
    ----------
    corpus : str or list
        collection of documents for training, allowed corpus are line sentence text files or list of tokens
    model : str
        neural embedding architecture, allowed values are 'Word2Vec' or 'FastText'
    hyperparameters : dict
        dictionary with mandatory keys for training configuration 'vector_size', 'window', 'min_count' and 'epochs'

    Methods
    -------
    Train()
        Starts the training procedure
    SaveModel(file_format: str {'txt', 'pickle'}, filename: str)
        Saves the output model, with all the collateral files needed
    WordVectors()
        Saves the *.wordvectors file for faster vector-query executions
    """


    def __str__(self):
        
        return "NeuralEmbeddingModels class"

    
    def __init__(self, corpus: str or list, model: str, hyperparameters: dict):
        
        arguments = locals()
        self.arguments = arguments
        
        # Read corpus
        try:
            os.path.isfile(arguments['corpus'])
            if os.path.isfile(arguments['corpus']) and arguments['model'] and arguments['hyperparameters']:
                pass
            else:
                raise Exception("Please provide one value for the arguments 'corpus', 'model' and 'hyperparameters'!")

        except:
            if type(arguments['corpus']) == list and arguments['model'] and arguments['hyperparameters']:
                pass
            else:
                raise Exception("Please provide one value for the arguments 'corpus', 'model' and 'hyperparameters'!")


    def Train(self):

        try:
            hparam = self.arguments['hyperparameters']
            print("\n...Training model")
            start = timeit.default_timer()
                
            # Train Word2Vec
            if self.arguments['model'].lower() == 'word2vec':
                model = Word2Vec(vector_size=hparam['vector_size'], window=hparam['window'], min_count=hparam['min_count'], 
                                workers=-1, sg=1, hs=0, negative=7, sample=1e-3)

            # Train FastText
            elif self.arguments['model'].lower() == 'fasttext':
                model = FastText(vector_size=hparam['vector_size'], window=hparam['window'], min_count=hparam['min_count'], 
                                workers=-1)
            else:
                raise ValueError(f"{self.arguments['model']} is NOT a valid argument, please provide one model in: ['Word2Vec', 'FastText']")
                
            if type(self.arguments['corpus']) == list:
                model.build_vocab(corpus_iterable=self.arguments['corpus'])
                model.train(corpus_iterable=self.arguments['corpus'], total_words=model.corpus_total_words, epochs=hparam['epochs'])
            elif os.path.isfile(self.arguments['corpus']):
                corpus_file = datapath(self.arguments['corpus'])
                model.build_vocab(corpus_file=corpus_file)
                model.train(corpus_file=corpus_file, total_words=model.corpus_total_words, epochs=hparam['epochs'])
            else:
                raise NameError(f"Corpus {self.arguments['corpus']} is NOT a valid argument, please provide a path to file or list object!")

            self.model = model
            end = timeit.default_timer()
            print(f"--> DONE {int((end - start))} second(s)!")
            
            return model

        except:
            print(format_exc())
            raise Exception


    def SaveModel(self, filename="TrainedNeuralEmbedding"):
        
        try:
            print("\n...Saving model")
            start = timeit.default_timer()
            
            self.model.save(f"{self.arguments['model'].lower()}_{filename}_{datetime.today().date()}.model")
            
            end = timeit.default_timer()
            print(f"--> DONE {int((end - start))} second(s)!")
            self.filename = filename

        except:
            print(format_exc())
            raise Exception


    def WordVectors(self):
        
        try:
            print("\n...Caching word vectors")
            start = timeit.default_timer()
            
            word_vectors = self.model.wv
            del self.model

            word_vectors.save(f"{self.arguments['model'].lower()}_{self.filename}_{datetime.today().date()}.wordvectors")
            
            end = timeit.default_timer()
            print(f"--> DONE {int((end - start))} second(s)!")

        except:
            print(format_exc())
            raise Exception
