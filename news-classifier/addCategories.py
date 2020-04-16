import pandas as pd
from news_classifier import predict_news_category

input_df = pd.read_csv('../all_data_refined.csv', encoding='utf-8')

print("hi")

categories = []
for i in range(len(input_df['id'])):
    if(isinstance(input_df.iloc[i]['text'],str)):
        temp = predict_news_category(input_df.iloc[i]['title'],input_df.iloc[i]['text'])
        categories.append(temp)
input_df['category'] = categories

print(input_df)
