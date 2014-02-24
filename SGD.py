import config
import numpy
import numpy.matlib
import random
import math
from TrainingInstance import *

VERBOSE = True

def initializeUnif(matrix, r):
		w,l  = matrix.shape
		for i in range(w):
			for j in range(l):
				matrix[i,j] = r*(random.random()*2-1) #recenter -1 to 1, then do -r to r 

def paramInit(LANG_SIZE):
	W = numpy.matlib.zeros((config.d, 2*config.d+1)) #TODO: ADD BIAS
	Ws = numpy.matlib.zeros((config.NUM_CLASSES, config.d))
	L = numpy.matlib.zeros((config.d, LANG_SIZE))

	initializeUnif(W,config.r)
	initializeUnif(Ws, config.r)
	initializeUnif(L,config.r)

	return W,Ws,L

def log(*args):
	if VERBOSE:
		print " ".join(str(item) for item in args)


def pos_neg(target, predProb):
	if target==2: #neutral
		d_total = 0
		d_num_wrong = 0

	else:
		prediction = predProb.argmax()

		d_total = 1
		if target < 2 and prediction < 2:
			d_num_wrong = 0
		elif target > 2 and prediction > 2:
			d_num_wrong = 0
		else:
			d_num_wrong = 1

	return d_num_wrong, d_total

	
def fine_grained(target, predProb):
	d_total = 1 
	if predProb.argmax() == target:
		d_num_wrong = 0
	else:
		d_num_wrong = 1
	return d_num_wrong, d_total

'''
This just looks at the test error at the root (ie full sentences)
'''
def fullError(test_inst, W, L, Ws, comp_fun):
	num_wrong = 0
	total = 0

	for inst in test_inst:
		error = inst.pushTotalError(W,L,Ws) #we don't care about the error return value
		if max(inst.tree.y)==numpy.inf or max(inst.tree.y)!=max(inst.tree.y): #have a nan, BAD!!!
				print "NaN or INf encountered, no bueno!!!"
				assert False

		d_num_wrong, d_total = comp_fun(inst.tree.score, inst.tree.y)
		num_wrong += d_num_wrong
		total += d_total

	print "full error: numwrong is %d, number total is %d" % (num_wrong, total)
	return num_wrong / float(total)


'''
Gets the test error over all nodes in the tree.
'''
def phraseError(test_inst, W,L,Ws, comp_fun):
	num_wrong = 0
	total = 0
	for inst in test_inst:
		error = inst.pushTotalError(W,L,Ws)
		for node in inst.parentFirstOrderingLeaves:
			if max(node.y)==numpy.inf or max(node.y)!=max(node.y): #have a nan, BAD!!!
				print "NaN or INf encountered, no bueno!!!"
				assert False
				
			d_num_wrong, d_total = comp_fun(node.score, node.y)
			num_wrong += d_num_wrong
			total += d_total

	print "phrase error: numwrong is %d, number total is %d" % (num_wrong, total)
	return num_wrong / float(total)


def getErrors(train_inst, test_inst, W, Ws,L):
	train_inst = train_inst[:config.max_est_train_error]
	
	errors = {}
	for phase in [("Train", train_inst), ("Test", test_inst)]:
		phase_name = phase[0]
		phase_insts = phase[1]
		errors[phase_name] = {}

		for error_type in [("Phrase", phraseError), ("Full", fullError)]:
			error_name = error_type[0]
			error_fxn = error_type[1]
			errors[phase_name][error_name] = {}

			for classify_type in [ ("FineGrained", fine_grained)]: #("PosNeg", pos_neg),
				classify_name = classify_type[0]
				classify_fxn = classify_type[1]

				err = error_fxn(phase_insts, W, L, Ws, classify_fxn)
				errors[phase_name][error_name][classify_name] = err 

	return errors

def printErrors(errors):
	for phase_name in ["Train", "Test"]:
		print ""
		for error_name in ["Phrase", "Full"]:
			for classify_name in ["FineGrained"]: #"PosNeg", 
				print phase_name, error_name, classify_name, ": ", 1 - errors[phase_name][error_name][classify_name] #accuracy levels




def updateLSparseGrad(L, gradLSparse, alpha, lambda_L):
	for j in gradLSparse:
		L[:,j]-= (alpha * gradLSparse[j] + lambda_L * L[:,j] )



'''
When developing, we will pass in dev_inst instead of test_inst.
'''
def runSGD(training_instances, test_instances, LANG_SIZE, lambda_reg, lambda_L, verbose=True):
	VERBOSE = verbose 
	W, Ws, L = paramInit(LANG_SIZE) 

	print "\n======================================\nRUNNING SGD"
	print "Training on %d training instances and %d testing instances" % (len(training_instances), len(test_instances))
	print "Regularization is Reg: %f L_Reg: %f" % (lambda_reg, lambda_L)
	if VERBOSE:
		init_errors = getErrors(training_instances, test_instances, W, Ws, L)
		log("Accuracy before training: ")
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

		learn_rate = config.alpha #can also divide by log(num_iters) to decrease learning rate over time.

		W = W - learn_rate*gradW - lambda_reg * W #gradually decrease the learning rate
		Ws = Ws - learn_rate*gradWs - lambda_reg * Ws
		updateLSparseGrad(L, gradLSparse, learn_rate, lambda_L)

		# W = W - config.alpha*gradW - lambda_reg * W
		# Ws = Ws - config.alpha*gradWs - lambda_reg * Ws
		# updateLSparseGrad(L, gradLSparse, config.alpha, lambda_L)


	final_errors = getErrors(training_instances, test_instances, W, Ws, L)
	print "Accuracy after training: "
	printErrors(final_errors)


	print "DONE RUNNING SGD\n======================================"
	return final_errors, W, Ws, L, errors_avg_log, errors_total_log




