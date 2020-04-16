import pandas as pd
import time
from datetime import datetime
from datetime import date
from datetime import timedelta
import matplotlib.pyplot as plt

#read csv file
def readCsvFile(path):
	df = pd.read_csv(path, encoding='utf-8')
	input_df = df.filter(['id', 'published', 'type', 'category'])
    return input_df

#cleaning published column
def cleanPublished(input_df):
	input_df = input_df[input_df['published'].notna()]
	timestamps = []
	for i in range(len(input_df['published'])):
		if(" " in input_df.iloc[i]['published']):
			d,t = input_df.iloc[i]['published'].split(" ")
			timestamps.append(d)  
		else:
			timestamps.append(input_df.iloc[i]['published']) 
	input_df['published'] = timestamps
    return input_df

# returns DF with articles within window
def lookbackWindowDF(windowSize, currentDate, input_df):
    current = datetime.strptime(currentDate,"%Y-%m-%d")
    oldest = current - timedelta(days=windowSize)
    ids, published, types, category = [],[],[],[]
    output_df = pd.DataFrame()
    for i in range(len(input_df['published'])):
        sample = datetime.strptime(input_df.iloc[i]['published'],"%Y-%m-%d")
        if(sample >= oldest and sample <= current):
            ids.append(input_df.iloc[i]['id'])
            published.append(input_df.iloc[i]['published'])
            types.append(input_df.iloc[i]['type'])
            category.append(input_df.iloc[i]['category'])
    output_df['id']=ids
    output_df['published']=published
    output_df['type']=types
    output_df['category']=category
    return output_df

# returns fraction of fake news in a particular category
def fakenessOfCategoryDF(input_df,categoryName):
    if(input_df.empty==True):
        return 0.0
    category_df = input_df[input_df['category']==categoryName]
    if(category_df.empty==True):
        return 0.0
    fake_df = category_df[category_df['type']=='fake']
    if(fake_df.empty == True):
        return 0.0
    categoryCount, temp1 = category_df.shape
    fake_count, temp2 = fake_df.shape
    return fake_count/categoryCount

# returns the fraction of news which fall under particular category
def categoryFractionDF(input_df, categoryName):
    if(input_df.empty==True):
        return 0.0
    category_df = input_df[input_df['category']==categoryName]
    if(category_df.empty == True):
        return 0.0
    categoryCount, temp1 = category_df.shape
    totalCount, temp2 = input_df.shape
    return categoryCount/totalCount

# predecessor of lookbackWindowDF method (for reference only)
def lookbackWindow(windowSize, currentDate, input_df, columnName):
    current = datetime.strptime(currentDate,"%Y-%m-%d")
    oldest = current - timedelta(days=windowSize)
    trending = []
    for i in range(len(input_df['published'])):
        sample = datetime.strptime(input_df.iloc[i]['published'],"%Y-%m-%d")
        if(sample >= oldest and sample <= current):
            trending.append(input_df.iloc[i][columnName])
    return trending

# predecessor of fakenessOfCategoryDF method (for reference only)
def fakenessOfCategory(windowSize, currentDate, input_df, categoryName):
    current = datetime.strptime(currentDate,"%Y-%m-%d")
    oldest = current - timedelta(days=windowSize)
    fakenessFraction = 0
    totalCount = 0
    for i in range(len(input_df['published'])):
        sample = datetime.strptime(input_df.iloc[i]['published'],"%Y-%m-%d")
        if(sample >= oldest and sample < current and input_df.iloc[i]['category']==categoryName):
            totalCount += 1
            if(input_df.iloc[i]['type']=='fake'):
                fakenessFraction += 1
    if(totalCount==0):
        return totalCount
    return fakenessFraction/totalCount

# for calculating fractions and identifying trending category.
def categoryFraction(trending):
    totalCount = len(trending)
    fractionDict = {
        'business':0,
        'entertainment':0,
        'politics':0,
        'sport':0,
        'tech':0,
        'others':0
    }
    if(totalCount==0):
        return fractionDict
    for i in trending:
        if i in fractionDict.keys():
            fractionDict[i] += 1
        else:
            fractionDict['others'] += 1
    for j in fractionDict.keys():
        fractionDict[j] /= totalCount
    return fractionDict

def fakenessFraction(trending):
    totalCount = len(trending)
    fractionDict = {
        'real':0,
        'fake':0
    }
    if(totalCount==0):
        return fractionDict
    for i in trending:
        fractionDict[i] += 1
    for i in fractionDict.keys():
        fractionDict[i] /= totalCount
    return fractionDict

#returns the oldest and latest date of publish among articles present in dataset
def minmaxPublishedDate(input_df):
    minDate = input_df['published'].min()
    maxDate = input_df['published'].max()
    return [minDate, maxDate]

# Applying lookdownWindowDF on entire dataset for generating analysis
def slidingWindow(windowSize,minDate,maxDate,input_df,categoryName):
    minValue = datetime.strptime(minDate,"%Y-%m-%d")
    currentDate = maxDate
    current = datetime.strptime(currentDate,"%Y-%m-%d")
    limit = minValue + timedelta(days=windowSize)
    results = []
    while(current>=minValue):
        currentDate = datetime.strftime(current,"%Y-%m-%d")
        sub_df = lookbackWindowDF(windowSize, currentDate, input_df)
        fraction = categoryFractionDF(sub_df, categoryName)
        results.append([fraction,fakenessOfCategoryDF(sub_df, categoryName)])
        current = current - timedelta(days=windowSize)
    return results

#Plotting charts
def drawPlot(X,Y,plotType,windowSize,categoryName):
	fig, ax = plt.subplots()
	if(plotType=='scatter'):
		ax.scatter(X, Y, label='Trend')
	else:
		ax.plot(X, Y, label='Trend')
	# Add some text for labels, title and custom x-axis labels, etc.
	ax.set_xlabel('Trend Fraction')
	ax.set_ylabel('Fake Fraction')
	ax.set_title('Fake Fraction v/s Trend Fraction for '+categoryName+' category')
	ax.legend()
	fig.tight_layout()
	plt.show()
	filename = categoryName+plotType+str(windowSize)+'.png'
	fig.savefig(filename, dpi = 400)

#draw both scatter and line charts for all categories with different window sizes
def drawAllPlots(input_df):
    minDate,maxDate=minmaxPublishedDate(input_df)
    for i in ['business', 'politics', 'sport', 'entertainment', 'tech']:
        for j in [3,5,7,9]:
            points = slidingWindow(j,minDate,maxDate,input_df,i)
            X,Y=[],[]
            for point in points:
                X.append(point[0])
                Y.append(point[1])
            print(X)
            print(Y)
            drawPlot(X,Y,'scatter',j,i)
            drawPlot(X,Y,'line',j,i)
    
#driver
if __name__ == "__main__":
	path = '../Datasets/Working_Data/all_data_refined_v2.csv'
	input_df = readCsvFile(path)
    input_df = cleanPublished(input_df)
	currentDate = "2016-11-15"
	windowSize = 7
	categoryName = "politics"
	minDate,maxDate=minmaxPublishedDate(input_df)
	points = slidingWindow(windowSize,minDate,maxDate,input_df,categoryName)
	X,Y=[],[]
	for point in points:
		X.append(point[0])
		Y.append(point[1])
	plotType = 'scatter'
	drawPlot(X,Y,plotType,windowSize,categoryName)



