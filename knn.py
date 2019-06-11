import random
import math
import copy
import matplotlib.pyplot as plt

def intro():
	testFile = "AllTestDataAlex.csv"
	trainFile = "AllTrainDataAlex.csv"
	return (testFile, trainFile)

def parseFile(testFile, trainFile):
	try:
		fileTest = open(testFile, "r")
		fileTrain = open(trainFile, "r")
	except IOError:
		print("Open failed")
		return

	dataTest = fileTest.readlines()
	dataTrain = fileTrain.readlines()

	dataListTest = []
	dataListTrain = []

	for i in dataTest:
		i = i.strip()	
		parsed = i.split(",")
		parsed = [float(j) for j in parsed]
		dataListTest.append(parsed)

	for i in dataTrain:
		i = i.strip()	
		parsed = i.split(",")
		parsed = [float(j) for j in parsed]
		dataListTrain.append(parsed)

	fileTest.close()
	fileTrain.close()

	# count = 1
	# for i in dataListTrain:
	# 	print("{}: {}".format(count, i))
	# 	count += 1

	return (dataListTest, dataListTrain)

def splitFeatureData(dataList):
	featureSplit = []
	dataSplit = []

	#make it random so that we can get different results from the same
	#data we are testing
	# random.shuffle(dataList)

	for i in dataList:
		featureSplit.append(i[-1])
		dataSplit.append(i[:-1])

	# count = 1

	# for i in featureSplit:
	# 	print("{}- {}".format(count,i))
	# 	count+=1

	return(featureSplit, dataSplit)

#want to split the data into 80 and 20 % to test knn.
#the 80 will be the training set, and the 20 will be the testing set.
def eightyTwentySplit(dataSplit):
	
	trainingSet = []
	testingSet = []

	eighty = math.ceil(len(dataSplit) * .8)
	twenty = len(dataSplit) - eighty

	trainingSet = dataSplit[0:int(eighty)]
	testingSet = dataSplit[int(eighty):]

	return (trainingSet, testingSet)

#we use the LP norm to calculate distance.
#when p = 1. it is the manhattan distacne.
#when p = 2 we are using euclidean distance.
#in this example we will just p = 2
#equation is as follows:
# d_p(x, y) = SUMMATION(from i to n) |(xi)^p - (yi)^p|^(1/p)
def euclideanDistance(list1, list2):

	distance = 0.0

	for i, j in zip(list1, list2):
		distance += abs(i - j)**2

	distance = distance**(1./2)

	return distance

def knn_classifier(trainingSet, testingSet, trainingLabel, testingLabel):

	tmpList = []
	kdiffClasses = []

	#take 1 unseen data point and check it on all the of the points we trained.
	for i in range(len(testingSet)):
		distanceList = []
		#traversing through all the training set.. we need to find the distacne to all of the points
		for j in range(len(trainingSet)):
			# print "Passing in -", trainingSet[j]
			calulatedDistance = euclideanDistance(testingSet[i], trainingSet[j])
			#j is the index of the just calculated distance
			distanceList.append((calulatedDistance, j))

		#sort the distance list by ascending distances
		distanceList = sorted(distanceList, key = lambda tup: tup[0])
		#this is where we pick k of the nearest neighbors. (in this case k = 1)
		# tmpList = distanceList[:5]
		
		#need it to count what the current point's nearest neighbor is
		class1 = 0
		class2 = 0
		classGood = 0
		classBad = 0

		# count = 1
		# for i in distanceList:
		# 	print("{}: {}".format(count, i))
		# 	count +=1

		# if trainingLabel[distanceList[0][1]] == 1.0:
		# 	class1 += 1
		# else:
		# 	class2 += 1
		#chooisng k = 10
		for z in range(10):
			if trainingLabel[distanceList[z][1]] == 0.0:
				classBad += 1
				# print("Giving badd")
			else:
				classGood += 1
				# print("Giving Good")

		#appending the class label (either 1 or 2), then what point we are testing (i)
		# if class1 > class2:
		# 	kdiffClasses.append((1.0, i))
		# else:
		# 	kdiffClasses.append((2.0, i))

		if classBad > classGood:
			kdiffClasses.append((0.0, i))
			# print("Labeled as a 0 (bad)")
		else:
			kdiffClasses.append((1.0, i))
			# print("Labeled as a 1 (good)")

		#this breaks when we have checked all of the points
		if len(kdiffClasses) == len(testingLabel):
			break

	return kdiffClasses

def forwardPropagation(testingSet, trainingSet, trainingLabel, testingLabel):

	#stores all the percentages for forward propagation
	bestFeaturePercentList = []
	featureList = []
	flp = []
	#both of these lists have all the single columns.
	testOneColumn = getTestAttributes(testingSet)
	trainOneColumn = getTrainAttributes(trainingSet)

	#for visualizing the data
	number_of_features = []
	graph_percentage = []

	#used to calculate the highest percentage
	bestFeaturePercent = 0

	for i in range(len(testOneColumn)):
		result = knn_classifier(trainOneColumn[i], testOneColumn[i], trainingLabel, testingLabel)
		percent = checkCalculation(result, testingLabel)
		featureList.append(i)
		flp.append(percent)
		if percent > bestFeaturePercent:
			bestFeaturePercent = percent
			feature = i

	#bestFeatureList will hold all the best features in order when using forward propagation.
	bestFeatureList = []
	bestFeatureList.append(feature)

	# print featureList
	# print flp

	for i in range(len(bestFeatureList)):
		print("Using feature(s) {", bestFeatureList[i], "} accuracy is", flp[i])
	print("Feature set {", feature,"} was best, accuracy is", bestFeaturePercent, "%\n")


	#Defined at the beginning of function.
	bestFeaturePercentList.append(bestFeaturePercent)

	#used so we can test each combination of pairs without editing the master (bestFeatureList) list
	tmpBestFeatures = []
	featureList = []
	neededToPrintList = []
 	#keeps looping until the TmpBestFeatures (defined above) has reached all features and tried all combinations
	while len(tmpBestFeatures) != len(testOneColumn) - 1:
		#need to reset the percent after trying each iteration
		bestFeaturePercent = 0
		#defined above. tl;dr need a copy so we dont edit the master copy.
		tmpBestFeatures = copy.deepcopy(bestFeatureList)

		neededToPrintList.append(feature)
		needToPrintPercent = []

		for index in range(len(testOneColumn)):
			if index not in bestFeatureList:

				tmpBestFeatures.append(index)
				knnTestingSet = []
				knnTrainingSet = []

				for j in range(len(testingSet)):
					tmp = []
					for i in range(len(tmpBestFeatures)):
						tmp.append(testingSet[j][tmpBestFeatures[i]])
					knnTestingSet.append(tmp)

				for j in range(len(trainingSet)):
					tmp = []
					for i in range(len(tmpBestFeatures)):
						tmp.append(trainingSet[j][tmpBestFeatures[i]])
					knnTrainingSet.append(tmp)

				result = knn_classifier(knnTrainingSet, knnTestingSet, trainingLabel, testingLabel)
				percent = checkCalculation(result, testingLabel)

				featureList.append(index)

				neededToPrintList.append(index)
				needToPrintPercent.append(percent)

				if percent > bestFeaturePercent:
					bestFeaturePercent = percent
					feature = index
				del tmpBestFeatures[-1]

		bestFeatureList.append(feature)
		
		print("Using feature(s) {", bestFeatureList, "} accuracy is", bestFeaturePercent)
		print("Feature set {", feature,"} was best, accuracy is", bestFeaturePercent, "%\n")
		graph_percentage.append(bestFeaturePercent)
		number_of_features.append(len(bestFeatureList))

		# bestFeatureList.append(feature)
		# print bestFeatureList

		# return

		bestFeaturePercentList.append(bestFeaturePercent)

	# print bestFeatureList
	# print bestFeaturePercentList
	plt.xlabel("Number of Features")
	plt.ylabel("Accuracy in Percentage")
	plt.title("Number of Features vs Accuracy")
	plt.plot(number_of_features, graph_percentage)
	plt.show()

	highest = -1

	# for i in range(len(bestFeaturePercentList)):
	# 	if bestFeaturePercentList[i] > highest:
	# 		highest = bestFeaturePercentList[i]
	# 		index2 = i

	# for i in range(index2 + 1):
	# 	print "Feature:", bestFeatureList[i] + 1

def getTestAttributes(testingSet):

	rtnList = []
	tmpTest =[]
	for i in range(len(testingSet[0])):
		tmpTest = []
		for j in range(len(testingSet)):
			# for k in range(depth):
			# 	tmpTest.append([testingSet[j][i]])
			tmpTest.append([testingSet[j][i]])
			# tmpTest.append(testingSet[j][i])
			#tmpTest = []
		rtnList.append(tmpTest)
	return rtnList

def getTrainAttributes(trainSet):

	rtnList = []
	for i in range(len(trainSet[0])):
		tmpTest = []
		for j in range(len(trainSet)):
			tmpTest.append([trainSet[j][i]])
			# tmpTest.append(trainSet[j][i])
		rtnList.append(tmpTest)
	return rtnList

def checkCalculation(resultTuple, testingLabel):

	counter = 0.0

	for i, j in zip(resultTuple, testingLabel):
		if i[0] == j:
			counter += 1.0

	# print "Percent - ", (counter / len(resultTuple)) * 100, "%"

	return (counter / len(resultTuple)) * 100

def main():
	#returns the name of the file to open.
	testFile, trainFile = intro()
	# algorithmNum = 1

	dataListTest, dataListTrain = parseFile(testFile, trainFile)
	# dataList = parseFile()

	train_data_label, train_data_no_label = splitFeatureData(dataListTrain)

	trainingSet, testingSet = eightyTwentySplit(train_data_no_label)
	trainingLabel, testingLabel = eightyTwentySplit(train_data_label)

	##################################################################################
	#running it will all the features and not just 1
	resultTuple = knn_classifier(trainingSet, testingSet, trainingLabel, testingLabel)
	percentCorrect = checkCalculation(resultTuple, testingLabel)

	forwardPropagation(testingSet, trainingSet, trainingLabel, testingLabel)

	# print(resultTuple)
	print(percentCorrect)
	##################################################################################

if __name__ == "__main__":
	main()
