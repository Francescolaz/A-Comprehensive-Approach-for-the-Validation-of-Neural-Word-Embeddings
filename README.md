# A Comprehensive Approach for the Validation of Neural Word Embeddings

How do you know if your embedding model is actually learning something valuable? How do you get fast insights over the hyperparameter configurations to use? How can you adapt your model to several practical challenges? 
The Python classes contained in this repository had been primarily built to help you get some intial and quick answers to the most common *Data scientist's* problems when testing different models. Here, you would find a compact and flexible working environment to implement the full-stack *Natural Language Processing* (NLP) pipeline.
Note that, this repository should not replace an extensive and in-depth analysis for the learning problem to be solved, instead it should be intended as a guideline to be used for testing an emebdding model over some real-life scenarios.


## Description

The core aim of the repository is to incorporate the different stages of a NLP pipeline in a series of easy-to-implement sequential classes, all in a single working environment. The executable Python scripts are included in the *utils* folder, listed below:

- ```TextPreprocessing()```
- ```NeuralEmbeddingModels()```
- ```EmbeddingValidation()```

The **TextPreprocessing()** class can handle a training corpus, provided as text file or list of strings, to be cleaned and pre-processed according to the most common techniques (ex. lowercasing, tokenization, etc.). Then, the returned corpus can stored locally and passed over the ```NeuralEmbeddingModels()``` class as training corpus for one of the two neural embedding architectures available (*Word2Vec* and *FastText*), which could be stored locally with all the learned word embedding vectors. In this stage, the produced word representations could be evaluated using the *Embedding Validation* class on five comprehensive embedding validation tasks of both instrinsic and extrinsic evaluators:

- *Word Relatedness*
- *Word Categorization*
- *Word Analogy*
- *Sentiment Analysis*
- *Frequency-Rare Word Classification*


## Getting Started


### Virtual Environment

The Python version used to implement the code is the $3.9.5$ one. The lists with all the required libraries could be find inside the *requirements.txt* file. In alternative, the *.venv* folder with the Python interpreter is available at the following address:

https://drive.google.com/drive/folders/1QpF4gn8BXpwJTgTVgOjfWXjBUa3PtHXQ?usp=share_link

Then, it could be downloaded and accessed, so that the *python.exe* file could be set as Kernel path when executing the code, without any depedency issues or prior installation needed.


### Datasets

The datasets used for the *Word Relatedness*, *Word Categorization* and *Word Analogy* are included in the *Datasets* folder of this repository. Additionally, the training corpus in line-sentence format and the dataset for the *Sentiment Analysis* embedding validation task are available at the following address:

https://drive.google.com/drive/folders/1yN6QYqmoUypXz5EQaeJPfr-2LoWqxmv9?usp=share_link

Moreover, in the *Plots* folder of this repository, you could find some interesting graphs summurazing the statistical properties of some of the datasources used.


### Models

All the 24 trained neural embedding models are available at the following address:

https://drive.google.com/drive/folders/16KKK0zPf076ySkPT-8SOg_yF5m6to6nM?usp=share_link

The root folder structure follows the following hierarchy:

1. Number of experiment: *First* or *Second*
2. Text preprocessing scenario: *None*, *Soft* or *Hard*
3. Number of hyperparameter configurations: *1st* or *2nd*
4. Neural model architecture: *FastText* or *Word2Vec*

They could be downloaded and tested with various other configurations of the ```EmbeddingValidation()``` arguments.


### Examples

Inside the *Code Examples* folder you could find some practical examples of how to import the libraries and the three main scripts, execute each of them with a mock parameter configuration and have a look of the outputs generated. Moreover, they could be used as a starting point on real-world datasets for different uses cases. 


### Operational Statistics

Inside the *Operational Statistics* folder of the repository, you could find a series of Excel spreadsheet with:

- The *Running Times* of the three embedding validation-pipeline stages.
- The four *Hyperparameter Configurations* used to train the neural embedding models.
- The *Working Environment Parameters* for the five evaluation tasks.
- The *Scoring Performances* of the two experiments of the case study.
- The *Ranking Matrix* of the twenty-four models for each embedding validation task.
- The *Spearman Correlation Matrix* of the five evaluation tasks.