#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:35:27 2020

@author: virajdesai
"""


from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from textblob.classifiers import NaiveBayesClassifier
import pandas as pd
from sklearn.metrics import accuracy_score


class EmotionClassify:
    def __init__(self):


        self.df = pd.read_csv('DATA.csv')
        self.a = pd.Series(self.df['Field1'])
        print(self.a)
        self.b = pd.Series(self.df['SIT'])
        self.new_df = pd.DataFrame({'Text': self.b, 'Emotion': self.a})

        self.stop = set(stopwords.words('english'))  ## stores all the stopwords in the lexicon
        self.exclude = set(string.punctuation)  ## stores all the punctuations
        self.lemma = WordNetLemmatizer()

        ## lets create a list of all negative-words
        self.negative = ['not', 'neither', 'nor', 'but', 'however', 'although', 'nonetheless', 'despite', 'except',
                         'even though', 'yet']

        ## create a separate list to store texts and emotion
        self.em_list = []
        self.text_list = []

        ## create the training set
        self.train = []
        
        ##create a test set
        self.test = []
        
        self.trainSplit = 0.9
         # stores the summarized text in a list
        self.sum_text_list = []

        # the e-score list stores the e-score for each document
        self.e_score_dict = {}

        # call the driver function
        self.main()

    '''
    A function for cleaning up all the documents
    # removes stop words
    # removes punctuations
    # uses lemmatizer
    '''

    def clean(self, doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in self.stop if i not in self.negative])
        punc_free = "".join([ch for ch in stop_free if ch not in self.exclude])
        normalized = " ".join([self.lemma.lemmatize(word) for word in punc_free.split()])
        return normalized

    '''
    Function to iterate and clean up all texts
    '''

    def iterateClean(self):
        for i in range(self.df.shape[0]):
            self.new_df.loc[i]['Text'] = self.clean(self.new_df.loc[i]['Text'])

    '''
    Function to iterate and populate text list
    '''

    def iteratePopText(self):
        for i in range(self.new_df.shape[0]):
            self.text_list.append(self.new_df.loc[i]['Text'])

    '''
    Function to iterate and populate emotion list
    '''

    def iteratePopEmotion(self):
        for i in range(self.new_df.shape[0]):
            self.em_list.append(self.new_df.loc[i]['Emotion'])
        
    '''
    Function to create training set
    '''
        
    def createTrain(self):
        trainLength = (self.new_df.shape[0])*self.trainSplit
        for i in range(trainLength):
            self.train.append([self.text_list[i], self.em_list[i]])
        #print(len(self.train))
        
    
    '''
    Function to create test set
    '''
    
    def createTest(self):
        testLength = (self.new_df.shape[0])*(1-self.trainSplit)
        for i in range(testLength, self.new_df.shape[0]):
            self.test.append([self.text_list[i], self.em_list[i]])
        #print(len(self.test))
        
    '''
    Function to create model
    classify the query text
    and then summarize other texts
    classify them and return a dictionary containing the e-score for all documents
    '''
    
    def classifyTextTrain(self):
        cl = NaiveBayesClassifier(self.train)
        return cl
    
    def performance(self,cl):
        predictions = []
        for i in range(len(self.test)):
            predictions.append(cl.classify(self.test[i][0]))
        actuals = []
        for j in range(len(self.test)):
            actuals.append(self.test[j][1])
        
        #print(len(predictions))
        #print(len(actuals))
        #print(nltk.classify.accuracy(predictions, actuals)*100)
        print("Accuracy :" + str(accuracy_score(predictions, actuals)*100))
        
    
    '''
    A function which is the driver for the entire class

    '''
    def main(self):
        self.iterateClean()
        self.iteratePopEmotion()
        self.iteratePopText()
        self.createTrain()
        self.createTest()
        cc = self.classifyTextTrain()
        self.performance(cc)

#method for predicting class for test data
def classifyText(cl, content):
     result = cl.prob_classify(content)
     for label in result.samples():
            print("%s: %f" % (label, result.prob(label)))
        

obj = EmotionClassify()
cl = obj.classifyTextTrain()  # trained Naive Bayes Classifier

classifyText(cl, "I love reading!")

#Due to an accuracy of approximately 50% we need a better classifier.
