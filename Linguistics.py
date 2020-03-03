#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 20:31:59 2020

@author: virajdesai
"""

def wordCount(content):
    wordCount,outside,inside = 0, 0, 1
    state = outside
    for i in content:
        if (i == ' ' or i == '\n' or i == '\t'):
            state = outside
        elif state == outside:
            state = inside
            wordCount += 1
    return wordCount

def averageWordCount(content):
    count = wordCount(content)
    sentences, outside, inside = 0, 0, 1
    state = outside
    for i in content:
        if (i == '.' or i == '!' or i == '?' or i == '...'):
            state = outside
        elif state == outside:
            state = inside
            sentences += 1
    averageWords = count/sentences
    return averageWords

def characterCount(content, character):
    count = 0
    for i in content:
        if i == character:
            count += 1
    return count

def exclamationMarkCount(content):
    return characterCount(content, '!')

def capitalLetterCount(content):
    totalCount = 0
    alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in alphabets:
        totalCount += characterCount(content, i)
    return totalCount
    
def questionMarkCount(content):
    return characterCount(content, '?')   
 

def wordify(content):
    return content.split(' ')

def wordwiseCount(content, finderWords):
    count = 0
    wordList = wordify(content)
    for i in wordList:
        if i in finderWords:
            count += 1
    return count

def negationCount(content):
    negations = ["do not", "don't", "does not", "doesn't", "am not", "are not", "aren't", "is not", "isn't", "did not", "didn't", "have not", "haven't", "had not", "hadn't", "should not", "shouldn't", "would not", "wouldn't", "will not", "won't"]
    return wordwiseCount(content.lower(), negations)

def firstPersonPronounCount(content):
    pronounsList = ["I", "me", "we", "us", "my", "mine", "our", "ours"]
    return wordwiseCount(content.lower(), pronounsList)
    

def displayLinguistics(content):
    wc = wordCount(content)
    print("Word count: " + str(wc))
    awc = averageWordCount(content)
    print("Average word count: " + str(awc))
    ec = exclamationMarkCount(content)
    print("Exclamation marks present: " + str(ec))
    cc = capitalLetterCount(content)
    print("Capital letters present: " + str(cc))
    qc = questionMarkCount(content)
    print("Question marks present: " + str(qc))
    nc = negationCount(content)
    print("Negations used: " + str(nc))
    fc = firstPersonPronounCount(content)
    print("First person pronouns present: " + str(fc))


st = '''Apart from counting words and characters, our online editor can help you to improve word choice and writing style, and, optionally, help you to detect grammar mistakes and plagiarism. To check word count, simply place your cursor into the text box above and start typing. You'll see the number of characters and words increase or decrease as you type, delete, and edit them. You can also copy and paste text from another program over into the online editor above. The Auto-Save feature will make sure you won't lose any changes while editing, even if you leave the site and come back later. Tip: Bookmark this page now.

Knowing the word count of a text can be important. For example, if an author has to write a minimum or maximum amount of words for an article, essay, report, story, book, paper, you name it. WordCounter will help to make sure its word count reaches a specific requirement or stays within a certain limit.

In addition, WordCounter shows you the top 10 keywords and keyword density of the article you're writing. This allows you to know which keywords you use how often and at what percentages. This can prevent you from over-using certain words or word combinations and check for best distribution of keywords in your writing.

In the Details overview you can see the average speaking and reading time for your text, while Reading Level is an indicator of the education level a person would need in order to understand the words you’re using.

Disclaimer: We strive to make our tools as accurate as possible but we cannot guarantee it will always be so.'''

displayLinguistics(st)