from TrainingInstance import *
import SGD
import config
import sys
import DataInit
import numpy
import numpy.matlib
import random



#Create Train, Dev,Test instances from file
results = DataInit.getInstances(config.max_train_inst, config.max_dev_inst, config.max_test_inst)
training_instances, dev_instances, test_instances, word_index, index_word = results
LANG_SIZE = len(word_index)





SGD.runSGD(training_instances, dev_instances, LANG_SIZE)


sys.exit(0)
'''==================================================================================='''
'''Everything below here is Stochastic Gradient Descent'''

def updateLSparseGrad(L, gradLSparse, alpha):
	for j in gradLSparse:
		L[:,j]-= alpha * gradLSparse[j]



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


#which test error metric to use-- phrase or full?
if config.TEST_METRIC == "full":
	test_error_metric = test_error_full
elif config.TEST_METRIC == "phrase":
	test_error_metric = test_error_phrase
else:
	assert False, 'invalid test metric' 

train_error_init = test_error_metric(training_instances[:config.max_est_train_error], W, L, Ws)
test_error_init = test_error_metric(test_instances, W, L, Ws)
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





