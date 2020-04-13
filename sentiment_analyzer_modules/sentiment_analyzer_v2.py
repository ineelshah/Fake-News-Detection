from textblob import TextBlob

def sentiment_scores(sentence):
    score = TextBlob(sentence)
    return score.sentiment

# Driver code 
if __name__ == "__main__" : 
    f = open('sentiment_test.txt','r')
    sentence = f.read()
    f.close()
    print(sentiment_scores(sentence))