Code for the Recursive Neural Net for Sentiment Analysis as described in the paper here: 
http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf

The rest of my code on github is here:
https://github.com/tcamenzi/NLP_Assignment.git


This section is more dry / detailed than the later ones; I would skip to the 
optimization / results section unless you want to see my error derivatives.
=======================================================
Related to TrainingInstance.py and taking the gradients with respect to things.

Notation:
t is the target vector for a node.
It is all 0s with a 1 in the kth index, where k is the correct sentiment.
y is the prediction vector, ie the probabilities of each one occurring. Dimension = 5 for 5 sentiment classes.
Ws is the prediction matrix; given the state of a node, it converts it to probabilities using
y = softmax(Ws*a)
Dimension of Ws: 5 x d
a:
a is the activation (or state) of a node. For the leaf nodes, it is simply the vector representation of the word;
its dimension is d.
f(...) will denote the vectorized tanh function, that maps input to the range (-1, 1) in an element-wise fashion.
W denotes the combination matrix; its dimension is dx2d.
a = f(W*[b;c;1]) where b,c  are the incoming (children) activations from the child nodes and '1' is the bias.
L is the vocabulary; each column is a d-vector, one column per word, that gives the representation of each word
in our language. We sometimes denote W*[b;c;1] as x, so a = f(x).

Documentation for Matrix Derivatives formulas:
The error for a node is -log(yk), where yk is the kth element of y; this is the KL divergence error. We want yk=1, yk=0 is unacceptabe.
The gradient of Ws is given by (y-t)*a.T where .T denotes transpose.

The softmax error is sigSoftmax; this is the error with respect to *x* coming in, not the activation a.
sigSoftmax = Ws.T(y-t) .* f'(x)
which is the same as
Ws.T(y-t) .* (1-aXXa)
where .* is element-wise multiplication.

The gradient with respect to W is
[ sigSoftmax + sigParent .* f'(x) ]*[b;c].T   where b,c are the activations of the left, right children.
summed over all non-leaf nodes.

sigDown = W.T * sigSoftmax + W.T * (sigParent .* f'(x))
Then the left half of sigDown becomes sigParent for the left child, and the right half of sigDown becomes sigParent for the right child.
Note sigParent = 0 for the root.

gradL:
Ws.T*(y-t) + sigParent

Note about L:
- L stores the direct activations of the words, ie we use the ones in L directly *without* doing tanh(L) first.


When working the the training instances, make sure things get done in the following order:
1) set activations a
2) set predictions y 
3) sum the errors (optional; requires y,t,a)
4) get grad wrt Ws (optional; requires y,t,a)
5) set softmax errors
6) set total errors
7) get grad wrt W (optional; requires all errors be set, Ws)
8) get grad wrt L (optional; requires all errors be set, Ws)

PushTotalError does steps 1,2,3 for you.

These derivations passed gradient check (accurate to .1% of each other)
so I have some confidence that the derivations are correct.

====================================================================
Related to SGD.py and optimization.

We use Stochastic Gradient Descent to optimize our function.

Initialization:
All parameters had values initialized uniformly randomly on the range (-r, r).

The vast majority of our parameters are in L; 22k words in the lexicon, d=25, means ~500k parameters in L.
In comparison, W and Ws will have (combined) <1000 parameters total, and these matrices contain the 
"intelligence" about how to combine words, negate phrases, etc. To keep things fast (even in python!),
we will update W, Ws each round using gradients, but do "sparse" updates to L. This means that we only
update the gradient with respect to words that appear in the current training instance (ie, ones that have
nonzero gradient).

For L's regularization term, even though we "should" update all terms each round, we only
regularize the ones whose words appear in that training instance. We have a separate regularization parameter
for L so that, if you want, you can decrease the values of L that *do* appear by a larger amount (since you only 
regularize them occasionally). Another option would be to, each round, randomly sample a subset of the parameter
values of L and regularize them.

Another note about regularization:
I found that very little regularization (or even no regularization) worked best; higher regularization parameters
hurt our accuracy on both the train and the test data. Overfitting did not seem to be a large problem, as the
train/test accuracy varied by only a few percent (usually 2-3%). This may be because there were relatively few
parameters outside of L to optimize. This was found using cross-validation on the dev set.

Another note about optimization:
We see few/minor decreases to our error function after a few thousand (2000-4000) iterations of SGD.
This means we have near-convergence before even going through all training instances!
It is possible the parameters in W, Ws are nearly converged, and the common words in L (your language) have converged.
Then running more iterations would help mainly because you gain better estimates for the rarer words in L.

Optimization was fast, and took ~2 minutes in Python.

====================================================
Results:
We obtain optimal results with the following parameters/settings.
d = 25
lambda_reg = lambda_L = .0001 #(small regularization coefficient)
alpha = .01 #learning rate
4000 iterations of SGD.

The fine-grained (5-class) classification accuracy over all nodes/phrases was
Train set: 91%
Dev set: 89%
Test set: 88%

This is great!

The fined-grained (5-class) classification accuracy for full sentences (only looking at the root node)
was terrible!
Train set: 14% accuracy
Dev set: 24%
Test set: 20%

In your paper you got ~40-45% accuracy for the root node classification.
I don't know what I did differently / wrong that lead to the lower accuracy rates.
However, I do know why the accuracy rates for the root are so low.

70% of all phrases are neutral, and the neural net is trying to minimize the error over 
all nodes. As a result, the neural net ends up assigning maximal probability to the neutral node
in 84% of all nodes. This helps reduce its overall error rate, and leads to the high accuracies
over all nodes.

However, only 15-20% of all root nodes are of neutral sentiment, yet the majority of predictions 
for the root node are still for neutral sentiment. This leads to the atrociously low accuracy
on these root nodes, yet a good overall error rate.

My attempted fix:
I tried updating the error function to give a significantly higher weight to the root node's error value
(ie give 10x weight to the root node) in an attempt to bring up this accuracy, and optimize for this new
error objective. Intuitively, this should cause the neural net to be more accurate at the root node, even
if this is somewhat at the cost of being accurate for the other nodes.

This did not work; in fact, it did not significantly change the train/test error on either the whole tree
or on the root. Two possible reasons:
1) Buggy implementation. 
2) It's hard to classify sentences correctly.... meaning that the larger derivates at the root tend to cancel
each other out (ie it optimizes the ones that are "easy" at the leaf nodes, and guesses at the root node because
it can't consisently get the root node right.) 

Let me know if you have other suggestions / explanations for the low root node error accuracy.


Observation: The vast majority of the parameters here are in L, learning the words for the language.
(Because some words are rare / not often updated, the number of effective / commonly seen parameters 
may be significantly less than 500k, but this is still much, much larger than the <1000 parameters in W and Ws).

Questions:
1) I assume one reason the tensor network performs so well is that it has more parameters (more "intelligence") in the
combining function, because it uses d^3 parameters instead of d^2 in the combination tensor matrix? In the RNN,
it seems that a lot of the training data is simply used to initialize the L-matrix, and that the combining function could support many more parameters (d=45, d=85 performed comparably to d=25).

2) Could you (somehow) initialize the words in L by training on unlabeled text data?
I know other neural nets have been trained to have generative models of text (ie by reading wikipedia) and can cluster
similar words together, can note different definitions of words based on the semantic context, etc. I assume that the deeper layers of the models would have a lower-dimensional representation of these words. Could you use these representations as a starting point for L, and then fine-tune these representations with the positive/negative labels?
Presumably, this would also let you perform well on words found in a large unlabeled training set, but not the smaller labeled training set of parse trees, assuming that the net could identify a similar word that is in the labeled training set and assume that the two words have the same positive/negative connotation.

Because L has so many parameters, it would be nice to learn it in an unsupervised manner, and save the supervised data
to learn smarter/more complex combination functions.