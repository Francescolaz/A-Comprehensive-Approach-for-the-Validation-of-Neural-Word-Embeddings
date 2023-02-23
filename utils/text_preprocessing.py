# Cleaning and pre-process training corpus


import os
import timeit
import io
import pickle
import string
import re
from nltk import WordNetLemmatizer
import spacy
import datetime
from datetime import datetime
from textblob import TextBlob
from nltk import pos_tag, word_tokenize
from traceback import format_exc
import joblib


class TextPreprocessing():


    """
    A class used to clean and pre-process text documents (or corpus)


    Attributes
    ----------
    digits : str or bool
        flag for dealing with digits, if True specify 'removal' or 'zero' (replace)
    lemma : str or bool
        flag for text lemmatization, if True specifiy one method among 'TextBlob', 'Spacy' or 'WordNet'
    POS_noun : str or bool
        flag for keeping only nouns, if True specifiy one method among 'TextBlob', 'Spacy' or 'WordNet'
    punctuation : bool
        flag for removing punctuation
    lowercase : bool
        flag for lowercasing text
    token : bool
        flag for text tokenization

    Methods
    -------
    FitText(corpus : str or None, list_text : str or None)
        Applies the selected pre-processing strategy to the provided text
    SaveText(file_format: str {'txt', 'pickle'}, filename: str)
        Saves the cleaned text as line sentece text file or pickle list of tokens
    """


    def __str__(self):
        
        return "TextPreprocessing class"


    def __init__(self, digits: str or bool, lemma: str or bool, POS_noun: str or bool, punctuation=True, lowercase=True, token=True):
        
        arguments = locals()
        self.arguments = arguments

    
    def FitText(self, corpus: str or None = None, list_text: str or None = None):
        
        try:
            print("\n...Reading corpus")
            start = timeit.default_timer()
            # Read corpus
            if corpus:
                if os.path.isfile(corpus):
                    with io.open(corpus, 'r', encoding="utf-8") as file:
                        corpus_list = file.readlines()
                        corpus_list = [line.rstrip() for line in corpus_list]
                else:
                    raise NameError("Please provide a valid string/path to file!")
            elif list_text:
                if os.path.isfile(corpus):
                    with open(list_text, 'rb') as f:
                        corpus_list = pickle.load(f)
                else:
                    raise NameError("Please provide a valid string/path to file!")
            else:
                raise NameError("Please provide a string/path to file OR a list of strings!")
            end = timeit.default_timer()  
            print(f"--> DONE {int((end - start))} second(s)!")
            
            start = timeit.default_timer()
            # Digits removal o zero replacement
            if (self.arguments['digits']) and (type(self.arguments['digits']) == str and self.arguments['digits'].lower() == 'removal'):
                print("\n...Removing digits")
                corpus_list = [i.translate({ord(k): None for k in string.digits}) for i in corpus_list]
                end = timeit.default_timer()
                print(f"--> DONE {int((end - start))} second(s)!")
            elif (self.arguments['digits']) and (type(self.arguments['digits']) == str and self.arguments['digits'].lower() == 'zero'):
                print("\n...Replacing digits with zero")
                corpus_list = [i.translate({ord(k): '0' for k in string.digits}) for i in corpus_list]
                end = timeit.default_timer()
                print(f"--> DONE {int((end - start))} second(s)!")
            elif self.arguments['digits']:
                raise ValueError(f"Please provide one value for 'digits' among: ['Removal', 'Zero']")
            else:
                pass

            start = timeit.default_timer()
            # Punctuation removal
            if self.arguments['punctuation']:
                print("\n...Removing punctuation")
                corpus_list = [i.translate(str.maketrans('', '', string.punctuation)) for i in corpus_list]
                end = timeit.default_timer()
                print(f"--> DONE {int((end - start))} second(s)!")
            
            start = timeit.default_timer()
            # Lowercasing
            if self.arguments['lowercase']:
                print("\n...Lowercasing text")
                corpus_list = [i.lower() for i in corpus_list]
                end = timeit.default_timer()
                print(f"--> DONE {int((end - start))} second(s)!")

            start = timeit.default_timer()
            # Tokenization
            if self.arguments['token']:
                print("\n...Tokenizing text")
                corpus_list = [re.compile(r'\w+').findall(i) for i in corpus_list]
                end = timeit.default_timer()
                print(f"--> DONE {int((end - start))} second(s)!")
                
            start = timeit.default_timer()
            # Lemmatization
            if self.arguments['lemma'] and self.arguments['lemma'].lower() == 'wordnet':
                print("\n...Lemmatizing text with WordNet")
                wn = WordNetLemmatizer()
                corpus_list = [' '.join([wn.lemmatize(w) for w in i]) for i in corpus_list]
                end = timeit.default_timer()
                print(f"--> DONE {int((end - start))} second(s)!")
            elif self.arguments['lemma'] and self.arguments['lemma'].lower() == 'spacy':
                print("\n...Lemmatizing text with Spacy")
                lem_spacy = spacy.load('en', disable=['parser', 'ner'])
                corpus_list = [" ".join([w.lemma_ for w in lem_spacy(i)]) for i in corpus_list]
                end = timeit.default_timer()
                print(f"--> DONE {int((end - start))} second(s)!")
            elif self.arguments['lemma'] and self.arguments['lemma'].lower() == 'textblob':
                print("\n...Lemmatizing text with TextBlob")
                corpus_list = [" ".join([w.lemmatize() for w in TextBlob(i).words]) for i in corpus_list]
                end = timeit.default_timer()
                print(f"--> DONE {int((end - start))} second(s)!")
            elif self.arguments['lemma']:
                raise Exception(f"Please provide one value for 'lemma' among: ['WordNet', 'Spacy', 'TextBlob']")
            else:
                pass

            start = timeit.default_timer()
            # Extract nouns with POS tagging
            if self.arguments['lemma'] and self.arguments['POS_noun'].lower() == 'wordnet':
                print("\n...Extracting nouns with WordNet")
                corpus_list = [" ".join([w for w, pos in pos_tag(word_tokenize(i)) if pos.startswith('N')]) for i in corpus_list]
                end = timeit.default_timer()
                print(f"--> DONE {int((end - start))} second(s)!")
            elif self.arguments['lemma'] and self.arguments['POS_noun'].lower() == 'spacy':
                print("...Extracting nouns with Spacy")
                pos_spacy = spacy.load('en_core_web_sm')
                corpus_list = ["".join([w for w in pos_spacy(i) if w.pos_ == 'NOUN']) for i in corpus_list]
                end = timeit.default_timer()
                print(f"--> DONE {int((end - start))} second(s)!")
            elif self.arguments['lemma'] and self.arguments['POS_noun'].lower() == 'textblob':
                print("\n...Extracting nouns with TextBlob")
                corpus_list = [" ".join([w for w, pos in TextBlob(i).tags if pos.startswith('NN')]) for i in corpus_list]
                end = timeit.default_timer()
                print(f"--> DONE {int((end - start))} second(s)!")
            elif self.arguments['lemma']:
                raise ValueError(f"Please provide one value for 'POS_noun' among: ['WordNet', 'Spacy', 'TextBlob']")
            else:
                pass

            self.corpus_list = corpus_list
            
            return corpus_list

        except:
            print(format_exc())
            raise Exception


    def SaveText(self, file_format: str, filename='PreProcessed-Text'):

        try:
            print("\n...Saving file")
            start = timeit.default_timer()
            # Write text file
            if file_format.lower() == 'txt':
                with io.open(f'{filename}_{datetime.today().date()}.{file_format}', 'w', encoding='utf-8') as f:
                    for line in self.corpus_list:
                        f.write(f"{line}\n")
                end = timeit.default_timer()
                print(f"--> DONE {int((end - start))} second(s)!")
                
            # Write pickle file
            elif file_format.lower() == 'pickle':
                with open(f'{filename}_{datetime.today().date()}', 'wb') as fp:
                    #pickle.dump(self.corpus_list, fp)
                    #pickle.dump([i.split(" ") for i in self.corpus_list], fp, pickle.HIGHEST_PROTOCOL)
                    joblib.dump([i.split(" ") for i in self.corpus_list], fp)
                end = timeit.default_timer()
                print(f"--> DONE {int((end - start))} second(s)!")
                
            else:
                raise ValueError("Please provide one value for 'file_format' among: ['Txt', 'Pickle']")

        except:
            print(format_exc())
            raise Exception
