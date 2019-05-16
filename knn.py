def parseFile(fileToOpen):
# def parseFile():
	try:
		# file = open("CS170_SMALLtestdata__30.txt", "r")
		# file = open("CS170_SMALLtestdata__30.txt", "r")
		file = open(fileToOpen, "r")
	except IOError:
		print "Open failed"
		return

	data = file.readlines()

	dataList = []

	for i in data:
		parsed = i.split(" ")
		parsed = [float(j) for j in parsed]
		dataList.append(parsed)

	file.close()

	return dataList

#splitting the data from its feature. It will correspond accordingly. i.e
# 1.0000000e+00 8.3624403e-01 1.4527562e+00 1.4174702e+00 -4.0407611e-01 7.3276154e-02 -1.7618845e+00 -3.7207882e-01 1.5982776e+00 7.9518550e-01 3.7655497e-01
# will be split into:
# featureSplit = 1.0
# dataSplit = 8.3624403e-01 1.4527562e+00 1.4174702e+00 -4.0407611e-01 7.3276154e-02 -1.7618845e+00 -3.7207882e-01 1.5982776e+00 7.9518550e-01 3.7655497e-01
# but there will be more because there is more than 1 object.
def splitFeatureData(dataList):
	featureSplit = []
	dataSplit = []

	#make it random so that we can get different results from the same
	#data we are testing
	# random.shuffle(dataList)

	for i in dataList:
		featureSplit.append(i[0])
		dataSplit.append(i[1:])

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

		# print tmpList

		if trainingLabel[distanceList[0][1]] == 1.0:
			class1 += 1
		else:
			class2 += 1

		#appending the class label (either 1 or 2), then what point we are testing (i)
		if class1 > class2:
			kdiffClasses.append((1.0, i))
		else:
			kdiffClasses.append((2.0, i))

		#this breaks when we have checked all of the points
		if len(kdiffClasses) == len(testingLabel):
			break

	return kdiffClasses

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
	fileToOpen, algorithmNum = intro()
	# algorithmNum = 1

	dataList = parseFile(fileToOpen)
	# dataList = parseFile()

	classLabels, dataSplit = splitFeatureData(dataList)

	trainingSet, testingSet = eightyTwentySplit(dataSplit)
	trainingLabel, testingLabel = eightyTwentySplit(classLabels)

	##################################################################################
	#running it will all the features and not just 1
	resultTuple = knn_classifier(trainingSet, testingSet, trainingLabel, testingLabel)
	percentCorrect = checkCalculation(resultTuple, testingLabel)
	##################################################################################
	if algorithmNum == 1:
		forwardPropagation(testingSet, trainingSet, trainingLabel, testingLabel)
	elif algorithmNum == 2:
		backPropagation(testingSet, trainingSet, trainingLabel, testingLabel)
	elif algorithmNum == 3:
		print "Alex's speical Algorithm (Alpha Beta Pruning)"
	else:
		print "Did not recgonize input"

if __name__ == "__main__":
	main()
	