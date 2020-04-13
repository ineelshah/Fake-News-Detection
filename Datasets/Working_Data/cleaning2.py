import pandas as pd
import re
from emotion_predictor import EmotionPredictor

def capitalWordCount(content):
    capitalCount = 0
    regex = '\s+[A-Z][A-Z]+\s|^[A-Z][A-Z]+.|\s+[A-Z][A-Z]+.|^[A-Z][A-Z]+:|\s+[A-Z][A-Z]+:|^[A-Z][A-Z]+,|\s+[A-Z][A-Z]+,'
    words = re.findall(regex, content)
    capitalCount = len(words)
    return capitalCount

input_df = pd.read_csv('all_data_refined.csv', encoding='utf-8')

actuals = []
articles = []
cap_words = []
for i in range(len(input_df['type'])):
	temp = capitalWordCount(input_df.iloc[i]['text'])
	cap_words.append(temp)
	if(isinstance(input_df.iloc[i]['text'],str)):
		articles.append(input_df.iloc[i]['text'])
		actuals.append(input_df.iloc[i]['type'])
input_df['capital_words'] = cap_words

model = EmotionPredictor(classification='plutchik', setting='mc', use_unison_model=True)

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

input_df.to_csv(r'all_data_refined.csv', index = False)

