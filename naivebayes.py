import csv


#separate each feature as a list of characters
def loadData(file):
	line = csv.reader(open(file))
	data = list(line)
	for x in range(len(data)):
		data[x] = [y.split() for y in data[x]]
	return data

traindatafile = 'TrainData - Druid.csv'
traindataset = loadData(traindatafile)
testdatafile = 'TestData - Druid.csv'
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
	dictlist = [dict() for x in range(7)]
	count = [0,0,0,0,0,0,0]
	for data in traindata:
		for x in range(len(data)-1):
			if data[x][0] not in dictlist[x]:
				dictlist[x][data[x][0]] = 0
				count[x]+=1
	return count
laplacesmoothing(traindataset)
def probabilityForGood(unseparateddata, traindata, card):
	probability = 1
	for x in range(len(card)):
		tempfreq = 0
		for y in traindata['1']:
			if card[x] == y[x]:
				tempfreq+=1
		probability *= float(tempfreq+1)/(len(traindata['1'])+laplacesmoothing(unseparateddata)[x])
	probability*= float(len(traindata['1']))/(len(traindata['1'])+len(traindata['0']))
	return probability


def probabilityForBad(unseparateddata, traindata, card):
	probability = 1
	for x in range(len(card)):
		tempfreq = 0
		for y in traindata['0']:
			if card[x] == y[x]:
				tempfreq+=1
		probability *= float(tempfreq+1)/(len(traindata['0'])+laplacesmoothing(unseparateddata)[x])
	probability*= float(len(traindata['0']))/(len(traindata['1'])+len(traindata['0']))
	return probability


def classify(unseparated, traindata, testdata):
	for x in testdata:
		print(x)
		good = probabilityForGood(unseparated, traindata, x)
		bad = probabilityForBad(unseparated, traindata, x)
		if good > bad:
			print("good")
		if bad > good:
			print("bad")
	return

classify(traindataset, separated, testdataset)