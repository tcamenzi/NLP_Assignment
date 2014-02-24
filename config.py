lambda_reg = .0001 #the regularization parameter for W, Ws
lambda_L = .0001 #the regularization parameter for L (because L is updated sparsely we have a separate parameter).

NUM_CLASSES = 5
d = 25 #the dimension of the word vector. About 25-35 is good.
r = .001 #uniform random initialization; should be small, ie .001 or so
USE_BABY = False #use a subset of the train/test set when debugging, so it fails faster.
max_iters = 4000 #maximum number of iterations to use for Stochastic Gradient Descent, 4k has been good.
root_x_factor = 1 #give the root more weight so we classify it correctly more often!!

if USE_BABY:
	max_iters = 400 #faster

alpha = .01 #the learning rate; .01 is good
max_est_train_error = 100 #use a subset of the training set to get the train error, otherwise too slow


INF = 99999999 #to use all the available data
max_train_inst = INF #len(training_instances)
max_dev_inst = INF 
max_test_inst = INF #len(test_instances) #100

