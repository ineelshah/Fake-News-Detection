from TemporalInterpreter import enhancer
import pandas as pd

def loadData(path):
	input_df = enhancer.readCsvFile(path)
	return input_df

def processing(input_df, instance_df, windowSize):
	input_df = enhancer.cleanPublished(input_df)
	instance_df = enhancer.cleanPublished(instance_df)
	currentDate = instance_df.iloc[0]['published']
	categoryName = instance_df.iloc[0]['category']
	actual_df = enhancer.lookbackWindowDF(windowSize, currentDate, input_df)
	return [input_df,instance_df,actual_df]

def calculateDelta(actual_df, windowSize, correlationMatrix, trendThreshold, categoryName):
	trendFraction = enhancer.categoryFractionDF(actual_df, categoryName)
	if(trendFraction<trendThreshold):
		return 0.0
	correlation = correlationMatrix[actual_df.iloc[0]['category']][windowSize]
	if(correlation>=0.2):
		delta = trendFraction*(correlation-0.3)/5
	elif(correlation<=-0.3):
		delta = trendFraction*(correlation+0.3)/5
	else:
		delta = 0.0
	return delta

def newRanges(originalRanges, delta):
	if(delta==0.0):
		return originalRanges
	for i in originalRanges.keys():
		if i == 'mostlyFake':
			originalRanges[i][1]+=delta
		elif i == 'mostlyReal':
			originalRanges[i][0]+=delta
		else:
			originalRanges[i][0]+=delta
			originalRanges[i][1]+=delta
	return originalRanges

def findResults(instance_df, windowSize):
	correlationMatrix = {
		'business':{3:0.082, 5:0.021, 7:0.203, 9:0.367},
		'politics':{3:-0.938, 5:-0.941, 7:-0.943, 9:-0.941},
		'entertainment':{3:-0.081, 5:0.176, 7:-0.248, 9:-0.565},
		'sport':{3:0.783, 5:0.795, 7:0.964, 9:0.903},
		'tech':{3:0.887, 5:0.901, 7:0.826, 9:0.861}
	}
	originalRanges={
		'mostlyFake':[0.0,0.20],
		'likelyFake':[0.20,0.40],
		'neutral':[0.40,0.60],
		'likelyReal':[0.60,0.80],
		'mostlyReal':[0.80,1.00]
	}
	datasetPath = '../Datasets/Working_Data/all_data_refined_v5.csv'
	input_df = loadData(datasetPath)
	trendThreshold = 0.3
	input_df, instance_df, actual_df = processing(input_df,instance_df,windowSize)
	categoryName = instance_df.iloc[0]['category']
	delta = calculateDelta(actual_df,windowSize,correlationMatrix,trendThreshold, categoryName)
	ranges = newRanges(originalRanges, delta)
	# call function which returns the result of TI-CNN model
	result = 0.4 
	resultCategory = findCategory(ranges, result)
	return {'ti-cnn':result,'delta':delta,'resultCategory':resultCategory,'ranges':ranges}

def findCategory(ranges, result):
	for i in ranges.keys():
		if (result>ranges[i][0] and result<=ranges[i][1]):
			return i

#predict final result category
def predictCategory(instance_df, windowSize):
	results = findResults(instance_df, windowSize)
	return results['resultCategory']

#predict actual probability of trueness
def predictProbability(instance_df,windowSize):
	results = findResults(instance_df, windowSize)
	value = results['ti-cnn']+results['delta']
	if(value<-0.0):
		return 0.0
	elif(value>=1.0):
		return 1.0
	else:
		return value

if __name__ == "__main__":
	path = '../Datasets/Working_Data/sample.csv'
	windowSize = 7
	instance_df = loadData(path)
	print(predictProbability(instance_df,windowSize))
	print(predictCategory(instance_df,windowSize))




