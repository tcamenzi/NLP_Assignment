from TrainingInstance import *

import sys

DATA_PATH = "../baby_trees/baby_"
TRAIN_FILE = DATA_PATH+'train.txt'
DEV_FILE = DATA_PATH+'dev.txt'
TEST_FILE = DATA_PATH+'test.txt'

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
from config import *
r = 10 #uniform random initialization
LANG_SIZE = len(word_index)

W = numpy.matlib.zeros((d, 2*d)) #TODO: ADD BIAS
Ws = numpy.matlib.zeros((NUM_CLASSES, d))
L = numpy.matlib.zeros((d, LANG_SIZE))

def initializeUnif(matrix, r):
	w,l  = matrix.shape
	for i in range(w):
		for j in range(l):
			matrix[i,j] = r*(random.random()*2-1) #recenter -1 to 1, then do -r to r 

initializeUnif(W,r)
initializeUnif(Ws, r)
initializeUnif(L,r)


a = "(3 (2 The) (2 Rock))"
te = training_instances[1]  # TrainingInstance(a)
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
	WsGradApprox = numpy.matlib.zeros((NUM_CLASSES, d))
	Ws_up = Ws.copy()
	Ws_down = Ws.copy()
	
	for i in range(NUM_CLASSES):
		for j in range(d):
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
	WGradApprox = numpy.matlib.zeros((d, 2*d))
	W_up = W.copy()
	W_down = W.copy()
	
	for i in range(d):
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
		LGradSparseApprox[j] = numpy.matlib.zeros((d,1))

	for i in range(d):
		for j in LGradSparse:
			# print "\nL_updown, i=%d, j=%d" % (i,j)
			# print L_up[i,j]
			# print L_down[i,j]
			L_up[i,j]+=eps
			L_down[i,j]-=eps
			# print "after"
			# print L_up[i,j]
			# print L_down[i,j]


			error1 = te.pushTotalError(W,L_up,Ws)
			error2 = te.pushTotalError(W,L_down,Ws)
			# print "error1: ", error1
			# print "error2: ", error2
			result = (error1-error2)/(2*eps)
			# print "result: ", result
			LGradSparseApprox[j][i,0] = result

			L_up[i,j]-=eps
			L_down[i,j]+=eps

	return LGradSparse, LGradSparseApprox

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

# gradLSparse, gradLSparseApprox = gradCheckLSparse(te, W, L, Ws)
# for i in gradLSparse:
# 	print "LGrad, LGradApprox for word %s:" % index_word[i]
# 	print gradLSparse[i].T
# 	print gradLSparseApprox[i].T
# 	print (gradLSparse[i]-gradLSparseApprox[i]).T
