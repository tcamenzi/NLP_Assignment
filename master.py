from TrainingInstance import *
import SGD
import config
import sys
import DataInit
import numpy
import numpy.matlib
import random

#Author: Tom Camenzind
#Citations: Data, technique from Richard Socher's Treebank Analysis dataset / paper.

lambda_reg = config.lambda_reg
lambda_L = config.lambda_L

#Create Train, Dev,Test instances from file
temp = DataInit.getInstances(config.max_train_inst, config.max_dev_inst, config.max_test_inst)
training_instances, dev_instances, test_instances, word_index, index_word = temp
LANG_SIZE = len(word_index)


#run SGD on the data
errors, W, Ws, L, errors_avg_log, errors_total_log = SGD.runSGD(training_instances, dev_instances, LANG_SIZE, lambda_reg, lambda_L)
print "above was dev errors; below is test errors, d=%d, root_x_factor = %d " % (config.d, config.root_x_factor)
test_errors = SGD.getErrors(training_instances, test_instances, W, Ws, L)
SGD.printErrors(test_errors)


'''
Make plots of the error from SGD 
'''
import matplotlib.pyplot as plt 
def myplot(error_log, msg):
	plt.plot(errors_avg_log)
	plt.xlabel("Number of iterations")
	plt.ylabel("Error")
	plt.title(msg)
	plt.show()

'''
Plot averages of 10 instead of individual
to get a smoother curve.
'''
def smooth(errors):
	WINDOW = 100
	smoothed = []
	for i in range(len(errors)-WINDOW):
		smoothed.append( sum(errors[i:i+WINDOW]) / float(WINDOW)) #can obvs make faster if needed
	return smoothed

import winsound #So you can take a nap and have it beep when it's done training. Training takes ~3 minutes.
Freq = 2500 # Set Frequency To 2500 Hertz
Dur = 1000 # Set Duration To 1000 ms == 1 second
winsound.Beep(Freq,Dur)

myplot(smooth(errors_avg_log), "Average per-node error")
myplot(smooth(errors_total_log), "Total error per tree")
