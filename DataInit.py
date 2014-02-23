import config
from TrainingInstance import *

if config.USE_BABY:
	DATA_PATH = "../baby_trees/baby_"
else:
	DATA_PATH = "../trees/"

TRAIN_FILE = DATA_PATH+'train.txt'
DEV_FILE = DATA_PATH+'dev.txt'
TEST_FILE = DATA_PATH+'test.txt'

'''
Instantiates instances from a single file
'''
def readInstances(filename, max_inst):
	instances = []
	f = open(filename)
	count = 0
	for line in f.readlines():
		instances.append(TrainingInstance(line))
		count+=1
		if count >= max_inst:
			break

	f.close()
	return instances 


'''
Handles reading in the instances from file and 
creates a 1:1 word:index mapping 
'''
def getInstances(max_train, max_dev, max_test):

	print "reading train"
	training_instances = readInstances(TRAIN_FILE, max_train)
	print "reading dev"
	dev_instances = readInstances(DEV_FILE, max_dev)
	print "reading test"
	test_instances = readInstances(TEST_FILE, max_test)
	print "done getting instances"

	#Here we assign each word a unique id or index in the range [0,num_words)
	all_instances = [training_instances, dev_instances, test_instances]
	allwords = set([])
	for instance_set in all_instances:
		for instance in instance_set:
			for word in instance.words:
				allwords.add(word)

	print "There are %d words in the lexicon" % len(allwords)

	index_word = []
	for word in allwords:
		index_word.append(word)
	word_index = {}
	for i in range(len(index_word)):
		word_index[index_word[i]] = i

	print "done making indexes"

	for instance_set in all_instances:
		for instance in instance_set:
			for node in instance.parentFirstOrderingLeaves:
				node.addID(word_index)

	return training_instances, dev_instances, test_instances, word_index, index_word


