import config
import numpy
import numpy.matlib
import random
from TrainingInstance import *

VERBOSE = True

def initializeUnif(matrix, r):
		w,l  = matrix.shape
		for i in range(w):
			for j in range(l):
				matrix[i,j] = r*(random.random()*2-1) #recenter -1 to 1, then do -r to r 

def paramInit(LANG_SIZE):
	W = numpy.matlib.zeros((config.d, 2*config.d)) #TODO: ADD BIAS
	Ws = numpy.matlib.zeros((config.NUM_CLASSES, config.d))
	L = numpy.matlib.zeros((config.d, LANG_SIZE))

	initializeUnif(W,config.r)
	initializeUnif(Ws, config.r)
	initializeUnif(L,config.r)

	return W,Ws,L

def log(*args):
	if VERBOSE:
		print " ".join(str(item) for item in args)


'''
This just looks at the test error at the root (ie full sentences)
'''
def fullError(test_inst, W, L, Ws):
	num_wrong = 0

	for inst in test_inst:
		error = inst.pushTotalError(W,L,Ws) #we don't care about the error return value
		if max(inst.tree.y)==numpy.inf or max(inst.tree.y)!=max(inst.tree.y): #have a nan, BAD!!!
				print "NaN or INf encountered, no bueno!!!"
				assert False

		if max(inst.tree.y)!=inst.tree.y[inst.tree.score]: #highest probability correction is at the correct index given by node.score, at the root
			num_wrong +=1

	print "full error: numwrong is %d, number of instances is %d" % (num_wrong, len(test_inst))
	return num_wrong / float(len(test_inst))


'''
Gets the test error over all nodes in the tree.
'''
def phraseError(test_inst, W,L,Ws):
	num_wrong = 0
	num_total = 0
	for inst in test_inst:
		error = inst.pushTotalError(W,L,Ws)
		for node in inst.parentFirstOrderingLeaves:
			num_total+=1
			if max(node.y)==numpy.inf or max(node.y)!=max(node.y): #have a nan, BAD!!!
				print "NaN or INf encountered, no bueno!!!"
				assert False
				
			if max(node.y)!=node.y[node.score]:
				num_wrong+=1

	return num_wrong / float(num_total)


def getErrors(train_inst, test_inst, W, Ws,L):
	train_inst = train_inst[:config.max_est_train_error]
	
	errors = {}
	for phase in [("Train", train_inst), ("Test", test_inst)]:
		inst = phase[1]
		errors[phase[0]] = {}
		for error_type in [("Phrase", phraseError), ("Full", fullError)]:
			error_fxn = error_type[1]
			err = error_fxn(inst, W, L, Ws)
			errors[phase[0]][error_type[0]] = err 
	return errors

def printErrors(errors):
	for phase in ["Train", "Test"]:
		for error_type in ["Phrase", "Full"]:
			print phase, error_type, ": ", errors[phase][error_type]




def updateLSparseGrad(L, gradLSparse, alpha):
	for j in gradLSparse:
		L[:,j]-= alpha * gradLSparse[j]



'''
When developing, we will pass in dev_inst instead of test_inst.
'''
def runSGD(training_instances, test_instances, LANG_SIZE, verbose=True):
	VERBOSE = verbose 
	W, Ws, L = paramInit(LANG_SIZE) 

	print "\n======================================\nRUNNING SGD"
	print "Training on %d training instances and %d testing instances" % (len(training_instances), len(test_instances))

	if VERBOSE:
		init_errors = getErrors(training_instances, test_instances, W, Ws, L)
		log("Errors before training: ")
		printErrors(init_errors)



	errors_avg_log = [] #the average per-node error
	errors_total_log = [] #the total error per tree
	itercount = 0

	#Here is where the actual stochastic gradient descent occurs
	while itercount < config.max_iters:
		itercount+=1
		if VERBOSE and (itercount % 100 == 0):
			print "itercount: ", itercount 
			print "avg, total error past 100: ", sum(errors_avg_log[-100:])/100.0, sum(errors_total_log[-100:])/100.0
			# print "top left part of W, Ws, L: "
			# print "W: "
			# print W[1:4, 1:4]
			# print Ws[1:4, 1:4]
			# print L[1:4, 1:4]


		inst = training_instances[random.randrange(len(training_instances))] #random training instance
		error = inst.pushTotalError(W,L,Ws)
		inst.setSoftmaxErrors(Ws)
		inst.setTotalErrors(W)

		gradW = inst.getGradW()
		gradLSparse = inst.getGradLSparse(Ws)
		gradWs = inst.getGradWs()

		num_nodes = len(inst.parentFirstOrderingLeaves) 
		errors_avg_log.append(error / float(num_nodes)) #record the average per-node error
		errors_total_log.append(error)

		W = W - config.alpha*gradW
		Ws = Ws - config.alpha*gradWs
		updateLSparseGrad(L, gradLSparse, config.alpha)


	final_errors = getErrors(training_instances, test_instances, W, Ws, L)
	print "Errors after training: "
	printErrors(final_errors)

	print "DONE RUNNING SGD\n======================================"
	return final_errors, W, Ws, L




