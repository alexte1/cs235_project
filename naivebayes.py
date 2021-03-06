import csv
import numpy as np
import matplotlib.pyplot as plt

#separate each feature as a list of characters
def loadData(file):
	line = csv.reader(open(file))
	data = list(line)
	for x in range(len(data)):
		data[x] = [y.split() for y in data[x]]
	return data

traindatafile = 'Data/TrainData - All - nomanacost.csv'
traindataset = loadData(traindatafile)
testdatafile = 'Data/TestData - All - nomanacost.csv'
testdataset = loadData(testdatafile)


#separate the loaded data into their class results: good(1) or bad(0)

def classSeparation(data):
	separated = {}
	for x in range(len(data)):
		v = data[x]
		if (v[-1][0] not in separated):
			separated[v[-1][0]] = []
		separated[v[-1][0]].append(v)
	return separated

separated = classSeparation(traindataset)	


#IMPLEMENT LAPLACE SMOOTHING TO ACCOUNT FOR DIVIDING BY O

def laplacesmoothing(traindata):
	dictlist = [dict() for x in range(3)]
	count = [0,0,0,0,0,0,0]
	for data in traindata:
		for x in range(len(data)-2):
			if data[x][0] not in dictlist[x]:
				dictlist[x][data[x][0]] = 0
				count[x]+=1
	return count
 
 #probability of a card being good
def probabilityForGood(unseparateddata, traindata, card):
	probability = 1
	for x in range(len(card)-2):
		tempfreq = 0
		for y in traindata['1']:
			if card[x] == y[x]:
				tempfreq+=1
		probability *= float(tempfreq+1)/(len(traindata['1'])+laplacesmoothing(unseparateddata)[x])
	probability*= float(len(traindata['1']))/(len(traindata['1'])+len(traindata['0']))
	return probability

#probability of a card being bad
def probabilityForBad(unseparateddata, traindata, card):
	probability = 1
	for x in range(len(card)-2):
		tempfreq = 0
		for y in traindata['0']:
			if card[x] == y[x]:
				tempfreq+=1
		probability *= float(tempfreq+1)/(len(traindata['0'])+laplacesmoothing(unseparateddata)[x])
	probability*= float(len(traindata['0']))/(len(traindata['1'])+len(traindata['0']))
	return probability

#compare probabilities for all cards, output results into a csv file and also plot predicted results and the actual classification
def classify(unseparated, traindata, testdata):
	correct = 0
	ratingval = 0
	fullresults = []
	resultfile=open('Data/results.csv', 'w')
	csvwriter = csv.writer(resultfile)
	for x in testdata:
		resultforgraph = []
		result = []
		result.append(' '.join(x[len(x)-1]))
		#print(x[len(x)-1])
		good = probabilityForGood(unseparated, traindata, x)
		bad = probabilityForBad(unseparated, traindata, x)
		if good > bad:
			#print("good")
			result.append("predicted good")
			resultforgraph.append(1)
			#print(x[3][0])
			result.append(int(x[3][0]))
			resultforgraph.append(int(x[3][0]))
			ratingval = 1
			if (int(x[3][0])==ratingval):
				correct+=1
		if bad > good:
			#print("bad")
			result.append("predicted bad")
			#print(x[3][0])
			resultforgraph.append(0)
			result.append(int(x[3][0]))
			resultforgraph.append(int(x[3][0]))
			ratingval = 0
			if (int(x[3][0])==ratingval):
				correct+=1
		csvwriter.writerow(result)
		fullresults.append(resultforgraph)
	resultfile.close()
	print(float(correct)/135)
	grapharray = np.array(fullresults)
	xval= np.arange(135)
	f1 = plt.figure(1)
	plt.plot(xval, grapharray[:,0], 'bo', markersize=2)
	plt.title('Predicted Classification')
	plt.ylabel('classification: 1=good, 0=bad')
	plt.xlabel('card')
	f2 = plt.figure(2)
	plt.plot(xval, grapharray[:,1], 'ro', markersize=2)
	plt.title('Actual Classification')
	plt.ylabel('classification: 1=good, 0=bad')
	plt.xlabel('card')
	plt.show()
	return

classify(traindataset, separated, testdataset)