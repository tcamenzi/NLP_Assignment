Code for the Recursive Neural Net for Sentiment Analysis as described in the paper here: 
http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf


Notation:
t is the target vector for a node.
It is all 0s with a 1 in the kth index, where k is the correct sentiment.
y is the prediction vector, ie the probabilities of each one occurring. Dimension = 5 because 5 sentiment classes.
Ws is the prediction matrix; given the state of a node, it converts it to probabilities using
y = softmax(Ws*a)
Dimension of Ws: 5xd
a:
a is the activation (or state) of a node. For the leaf nodes, it is simply the vector representation of the word;
its dimension is d.
f(...) will denote the vectorized tanh function, that maps input to the range (-1, 1) in an element-wise fashion.
W denotes the combination matrix; its dimension is dx2d.
a = f(W*[b;c]) where b,c  are the incoming (children) activations from the child nodes.
L is the vocabulary; each column is a d-vector, one column per word, that gives the representation of each word
in our language. We sometimes denote W*[b;c] as x, so a  f(x).

Documentation for Matrix Derivatives formulas:
The error for a node is -log(yk), where yk is the kth element of y; this is the KL divergence error. We want yk=1, yk=0 is unacceptabe.
The gradient of Ws is given by (y-t)*a.T where .T denotes transpose.

The softmax error is sigSoftmax; this is the error with respect to *x* coming in, not the activation a.
sigSoftmax = Ws.T(y-t) XX f'(x)
which is the same as
Ws.T(y-t) XX (1-aXXa)
where XX is element-wise multiplication.

The gradient with respect to W is
[ sigSoftmax + sigParent XX f'(x) ]*[b;c].T   where b,c are the activations of the left, right children.
summed over all non-leaf nodes.

sigDown = W.T * sigSoftmax + W.T * (sigParent XX f'(x))
Then the left half of sigDown becomes sigParent for the left child, and the right half of sigDown becomes sigParent for the right child.
Note sigParent = 0 for the root.

gradL:
Ws.T*(y-t) + sigParent

Note about L:
- L stores the direct activations of the words, ie we use the ones in L directly *without* doing tanh(L) first.

