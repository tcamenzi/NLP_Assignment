from TrainingInstance import *
import config

import sys


if USE_BABY:
	DATA_PATH = "../baby_trees/baby_"
else:
	DATA_PATH = "../trees/"

TRAIN_FILE = DATA_PATH+'train.txt'
DEV_FILE = DATA_PATH+'dev.txt'
TEST_FILE = DATA_PATH+'test.txt'



'''===================================================================='''
'''Everything below here is initializing train/test/dev instances'''

def getInstances(filename):
	instances = []
	f = open(filename)
	for line  in f.readlines():
		instances.append(TrainingInstance(line))
	return instances 

print "reading train"
training_instances = getInstances(TRAIN_FILE)
print "reading dev"
dev_instances = getInstances(DEV_FILE)
print "reading test"
test_instances = getInstances(TEST_FILE)
print "done getting instances"

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

'''==============================================================================='''
import numpy
import numpy.matlib
import random

LANG_SIZE = len(word_index)

W = numpy.matlib.zeros((config.d, 2*config.d)) #TODO: ADD BIAS
Ws = numpy.matlib.zeros((config.NUM_CLASSES, d))
L = numpy.matlib.zeros((config.d, LANG_SIZE))

def initializeUnif(matrix, r):
	w,l  = matrix.shape
	for i in range(w):
		for j in range(l):
			matrix[i,j] = r*(random.random()*2-1) #recenter -1 to 1, then do -r to r 

initializeUnif(W,config.r)
initializeUnif(Ws, config.r)
initializeUnif(L,config.r)


a = "(3 (2 The) (2 Rock))"
te = training_instances[4]  # TrainingInstance(a)
for node in te.parentFirstOrderingLeaves:
	node.addID(word_index)

# id1 = word_index["The"]
# id2 = word_index["Rock"]
# L[:,id1] = numpy.matrix('1;0')
# L[:,id2] = numpy.matrix('0;1')
# W = numpy.matrix('1 0 2 5; 3 0 4 6').astype(float)
te.setActivations(W,L)
# Ws = numpy.matrix('1 0; 1 0; 1 0; 0 2; 0 2').astype(float)
te.setPredictions(Ws)
error = te.totalError()
print "error: ", error
print te.tree.activation

def gradCheckWs(te, W, L, Ws):
	error = te.pushTotalError(W,L,Ws) #this sets y,t,a and gives the error
	WsGrad = te.getGradWs()

	eps = .0001
	WsGradApprox = numpy.matlib.zeros((config.NUM_CLASSES, d))
	Ws_up = Ws.copy()
	Ws_down = Ws.copy()
	 
	for i in range(config.NUM_CLASSES):
		for j in range(config.d):
			Ws_up[i,j]+=eps
			Ws_down[i,j]-=eps

			error1 = te.pushTotalError(W,L,Ws_up)
			error2 = te.pushTotalError(W,L,Ws_down)
			result = (error1-error2)/(2*eps)
			WsGradApprox[i,j] = result

			Ws_up[i,j]-=eps
			Ws_down[i,j]+=eps

	return WsGrad, WsGradApprox

def gradCheckW(te, W, L, Ws):
	error = te.pushTotalError(W,L,Ws) #this sets y,t,a and gives the error
	WGrad = te.getGradW()

	eps = .0001
	WGradApprox = numpy.matlib.zeros((config.d, 2*config.d))
	W_up = W.copy()
	W_down = W.copy()
	
	for i in range(config.d):
		for j in range(2*d):
			W_up[i,j]+=eps
			W_down[i,j]-=eps

			error1 = te.pushTotalError(W_up,L,Ws)
			error2 = te.pushTotalError(W_down,L,Ws)
			result = (error1-error2)/(2*eps)
			WGradApprox[i,j] = result

			W_up[i,j]-=eps
			W_down[i,j]+=eps

	return WGrad, WGradApprox

def gradCheckLSparse(te, W, L, Ws):
	error = te.pushTotalError(W,L,Ws) #this sets y,t,a and gives the error
	LGradSparse = te.getGradLSparse(Ws)

	eps = .0001
	LGradSparseApprox = {}
	L_up = L.copy()
	L_down = L.copy()
	
	for j in LGradSparse:
		LGradSparseApprox[j] = numpy.matlib.zeros((config.d,1))

	for i in range(config.d):
		for j in LGradSparse:
			L_up[i,j]+=eps
			L_down[i,j]-=eps


			error1 = te.pushTotalError(W,L_up,Ws)
			error2 = te.pushTotalError(W,L_down,Ws)
			result = (error1-error2)/(2*eps)
			LGradSparseApprox[j][i,0] = result

			L_up[i,j]-=eps
			L_down[i,j]+=eps

	return LGradSparse, LGradSparseApprox

PRINT_GRADCHECK = False 
if PRINT_GRADCHECK:
	WsGrad, WsGradApprox = gradCheckWs(te, W,L,Ws)
	print "WsGrad: ", WsGrad
	print "WsGradApprox: ", WsGradApprox
	print "Difference: ", WsGrad - WsGradApprox

	te.setSoftmaxErrors(Ws)
	te.setTotalErrors(W)

	WGrad, WGradApprox = gradCheckW(te, W,L,Ws)
	print "WGrad: ", WGrad
	print "WGradApprox: ", WGradApprox
	print "Difference: ", WGrad - WGradApprox

	gradLSparse, gradLSparseApprox = gradCheckLSparse(te, W, L, Ws)
	for i in gradLSparse:
		print "LGrad, LGradApprox for word %s:" % index_word[i]
		print gradLSparse[i].T
		print gradLSparseApprox[i].T
		print (gradLSparse[i]-gradLSparseApprox[i]).T








'''==================================================================================='''
'''Everything below here is Stochastic Gradient Descent'''

def updateLSparseGrad(L, gradLSparse, alpha):
	for j in gradLSparse:
		L[:,j]-= alpha * gradLSparse[j]

'''
This just looks at the test error at the root (ie full sentences)
'''
def test_error_full(test_inst, W, L, Ws):
	num_wrong = 0

	for inst in test_inst:
		error = inst.pushTotalError(W,L,Ws) #we don't care about the error return value
		if max(inst.tree.y)!=inst.tree.y[inst.tree.score]: #highest probability correction is at the correct index given by node.score, at the root
			num_wrong +=1

	return num_wrong / float(len(test_inst))

'''
Gets the test error over all nodes in the tree.
'''
def test_error_phrase(test_inst, W,L,Ws):
	num_wrong = 0
	num_total = 0
	for inst in test_inst:
		error = inst.pushTotalError(W,L,Ws)
		for node in inst.parentFirstOrderingLeaves:
			num_total+=1
			if max(node.y)==numpy.inf or max(node.y)!=max(node.y): #have a nan, BAD!!!
				print "NaN or INf encountered, no bueno!!!"
			if max(node.y)!=node.y[node.score]:
				num_wrong+=1
	print "did a total of %d comparisions " % num_total
	return num_wrong / float(num_total)



initializeUnif(W,r)
initializeUnif(Ws, r)
initializeUnif(L,r)


errors_log = []
itercount = 0

training_instances = training_instances[:config.max_train_inst]

'''
While developing & finding parameters,
use the dev set not the test set!!
'''
if config.DEV_MODE:
	test_instances = dev_instances

test_instances = test_instances[:config.max_test_inst]


print "Training on a set of %d training instances " % len(training_instances)
print "Testing  on a set of %d testing  instances"  % len(test_instances)

train_error_init = test_error_phrase(training_instances[:config.max_est_train_error], W, L, Ws)
test_error_init = test_error_phrase(test_instances, W, L, Ws)
print "Initial train error, before any training, is %f" % train_error_init
print "Initial test error, before any training, is %f" %  test_error_init

while itercount < config.max_iters:
	itercount+=1
	if itercount % 100 == 0:
		print "itercount: ", itercount 
		print "avg error past 100: ", sum(errors_log[-100:])/100.0
		# print "top left part of W, Ws, L: "
		# print "W: "
		# print W[1:4, 1:4]
		# print Ws[1:4, 1:4]
		# print L[1:4, 1:4]


	inst = training_instances[random.randrange(len(training_instances))]
	error = inst.pushTotalError(W,L,Ws)
	inst.setSoftmaxErrors(Ws)
	inst.setTotalErrors(W)

	gradW = inst.getGradW()
	gradLSparse = inst.getGradLSparse(Ws)
	gradWs = inst.getGradWs()

	errors_log.append(error)
	W = W - config.alpha*gradW
	Ws = Ws - config.alpha*gradWs
	updateLSparseGrad(L, gradLSparse, config.alpha)

print "Final training set error, before training: %f\nafter training: %f" % (train_error_init, test_error_phrase(training_instances[:config.max_est_train_error], W, L, Ws))
print "Final error, before training:%f\nafter training: %f" % (test_error_init, test_error_phrase(test_instances, W, L, Ws))





