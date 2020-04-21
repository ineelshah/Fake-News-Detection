import pandas as pd
import re
from EmotionClassification import emotion_predictor

def capitalWordCount(content):
    capitalCount = 0
    regex = '\s+[A-Z][A-Z]+\s|^[A-Z][A-Z]+.|\s+[A-Z][A-Z]+.|^[A-Z][A-Z]+:|\s+[A-Z][A-Z]+:|^[A-Z][A-Z]+,|\s+[A-Z][A-Z]+,'
    words = re.findall(regex, content)
    capitalCount = len(words)
    return capitalCount

def addCapitalWordCount(input_df):
	cap_words = []
	for i in range(len(input_df['type'])):
		temp = capitalWordCount(input_df.iloc[i]['text'])
		cap_words.append(temp)

	input_df['capital_words'] = cap_words
	return input_df

def addEmotions(input_df):
	actuals = []
	articles = []
	for i in range(len(input_df['type'])):
		if(isinstance(input_df.iloc[i]['text'],str)):
			articles.append(input_df.iloc[i]['text'])
			actuals.append(input_df.iloc[i]['type'])

	model = emotion_predictor.EmotionPredictor(classification='plutchik', setting='mc', use_unison_model=True)

	predictions = model.predict_classes(articles)
	input_df['emotion'] = predictions['Emotion'].astype(str)

	probabilities = model.predict_probabilities(articles)
	input_df['anger'] = probabilities['Anger'].astype(float)
	input_df['disgust'] = probabilities['Disgust'].astype(float)
	input_df['fear'] = probabilities['Fear'].astype(float)
	input_df['joy'] = probabilities['Joy'].astype(float)
	input_df['sadness'] = probabilities['Sadness'].astype(float)
	input_df['surprise'] = probabilities['Surprise'].astype(float)
	input_df['trust'] = probabilities['Trust'].astype(float)
	input_df['anticipation'] = probabilities['Anticipation'].astype(float)
	return input_df

if __name__ == "__main__":
	input_df = pd.read_csv('all_data_refined.csv', encoding='utf-8')
	input_df = addCapitalWordCount(input_df)
	input_df = addEmotions(input_df)
	input_df.to_csv(r'all_data_refined.csv', index = False)
