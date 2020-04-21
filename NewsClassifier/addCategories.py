import pandas as pd
import news_classifier

def returnCategory(input_df):
	return news_classifier.predict_news_category(input_df.iloc[0]['title'],input_df.iloc[0]['text'])

def addCategories(input_df):
	categories = []
	for i in range(len(input_df['id'])):
		if(isinstance(input_df.iloc[i]['text'],str)):
			temp = news_classifier.predict_news_category(input_df.iloc[i]['title'],input_df.iloc[i]['text'])
			categories.append(temp)
	input_df['category'] = categories
	return input_df

if __name__ == "__main__":
	input_df = pd.read_csv('../Datasets/Working_Data/sample.csv', encoding='utf-8')
	print(returnCategory(input_df))
