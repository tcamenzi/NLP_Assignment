from TrainingInstance import *
import SGD
import config
import sys
import DataInit
import numpy
import numpy.matlib
import random

lambda_reg = config.lambda_reg
lambda_L = config.lambda_L

# lambda_reg = float(sys.argv[1])
# lambda_L = float(sys.argv[2])


#Create Train, Dev,Test instances from file
results = DataInit.getInstances(config.max_train_inst, config.max_dev_inst, config.max_test_inst)
training_instances, dev_instances, test_instances, word_index, index_word = results
LANG_SIZE = len(word_index)





errors, W, Ws, L = SGD.runSGD(training_instances, dev_instances, LANG_SIZE, lambda_reg, lambda_L)
print "returned errors: ", errors
