from Sentiment import sentiment_analyzer_v1
import pandas as pd

def sentenceCount(content):
	sentences, outside, inside = 0, 0, 1
	state = outside
	for i in content:
		if (i == '.' or i == '!' or i == '?' or i == '...'):
			state = outside
		elif state == outside:
			state = inside
			sentences += 1
	return sentences

def addsentiments(input_df):
	negative,positive,neutral=[],[],[]
	for j in range(len(input_df['text'])):
		temp = sentiment_analyzer_v1.sentiment_scores(input_df.iloc[j]['text'])
		negative.append(temp['neg'])
		neutral.append(temp['neu'])
		positive.append(temp['pos'])
	input_df['neg']=negative
	input_df['neu']=neutral
	input_df['pos']=positive
	return input_df

def addsentenceCount(input_df):
	sentences = []
	for i in range(len(input_df['text'])):
		temp = sentenceCount(input_df.iloc[i]['text'])
		sentences.append(temp)
	input_df['sentence_Count'] = sentences
	return input_df

def addSentimentCategory(input_df):
	sentimentCategory = []
	for i in range(len(input_df['text'])):
		if(input_df.iloc[i]['neg']>input_df.iloc[i]['pos']):
			sentimentCategory.append('negative')
		elif(input_df.iloc[i]['pos']>input_df.iloc[i]['neg']):
			sentimentCategory.append('positive')
		else:
			sentimentCategory.append('neutral')
	input_df['sentimentCategory'] = sentimentCategory
	return input_df

if __name__ == "__main__" : 
	input_df = pd.read_csv('../Datasets/Working_Data/all_data_refined_v4.csv', encoding='utf-8')
	input_df = addsentenceCount(input_df)
	input_df = addsentiments(input_df)
	input_df = addSentimentCategory(input_df)
	input_df.to_csv(r'../Datasets/Working_Data/all_data_refined_v3.csv', index = False)

