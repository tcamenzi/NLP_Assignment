#TEST_METRIC = "full" #valid options: "phrase" or "full"


#lambda_reg = .01
#lambda_L = .01

NUM_CLASSES = 5
d = 85 #the dimension of the word vector. About 25-35 is good.
r = .001 #uniform random initialization; should be small, ie .001 or so
USE_BABY = False #use a subset of the train/test set when debugging, so it fails faster.
max_iters = 4000 #maximum number of iterations to use for Stochastic Gradient Descent, 4k has been good.

if USE_BABY:
	max_iters = 400 #faster

alpha = .01 #the learning rate; .01 is good


max_est_train_error = 100 #use a subset of the training set to get the train error, otherwise too slow

DEV_MODE = True

INF = 99999999 #to use all the available data
max_train_inst = INF #len(training_instances)
max_dev_inst = INF 
max_test_inst = INF #len(test_instances) #100





'''
Copy of values for a goodrun:
Final training set error, before training: 0.785814
after training: 0.079795
did a total of 21274 comparisions 
Final error, before training:0.768638
after training: 0.115493

89% accuracy!!!
Note: this is for saying, at all nodes, is it right/not? So for a single node, just saying
"This word is positive" gives you something. Todo: check for whole sentence classificiation,
both 5-grained and 2-grained.

However, it does *not* perform that well for full + fine grained.


Params file snapshot
---------------------------------------------------------------------------
TEST_METRIC = "phrase"
NUM_CLASSES = 5
d = 25 #the dimension of the word vector. About 25-35 is good.
r = .001 #uniform random initialization; should be small, ie .001 or so
USE_BABY = False  #use a subset of the train/test set when debugging, so it fails faster.
max_iters = 4000 #maximum number of iterations to use for Stochastic Gradient Descent, 4k has been good.

if USE_BABY:
	max_iters = 400 #faster

alpha = .01 #the learning rate; .01 is good


max_est_train_error = 100

DEV_MODE = True

INF = 99999999 #to use all the available data
max_train_inst = INF #len(training_instances)
max_test_inst = INF #len(test_instances) #100
------------------------------------------------------------------------------
'''



