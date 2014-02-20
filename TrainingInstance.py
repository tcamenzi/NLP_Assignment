import numpy
import numpy.matlib
import math
from config import *

'''
Input: a matrix.
Output: Does softmax on each column of a matrix.
A matrix with 1 column does a normal softmax; ie 
softmax on [1;1;1] returns [.33;.33;.33]
'''
def softmax(matrix):
	e = numpy.exp(matrix)
	return e / sum(e)


class Node:
	def __init__(self):
		self.parent = None
		self.left = None
		self.right = None
		self.isLeaf = False
		self.score = None 
		self.t = None #for target; using the notation in the paper

	def setScore(self, score):
		self.score = score
		self.t = numpy.matlib.zeros((NUM_CLASSES, 1))
		self.t[score] = 1

	def makeLeaf(self, word):
		self.isLeaf = True
		self.word = word

	def addID(self, word_id_mapping):
		word_id = word_id_mapping[self.word]
		self.word_id = word_id 

class TrainingInstance:
	def __init__(self, instring):
		self.words = []
		self.tree = self.buildTree(instring)
		self.parentFirstOrdering = []
		self.setOrdering(self.tree)
		self.parentFirstOrderingLeaves = []
		self.parentFirstOrderingNonLeaves = []

		for node in self.parentFirstOrdering:
			if node.isLeaf:
				self.parentFirstOrderingLeaves.append(node)
			else:
				self.parentFirstOrderingNonLeaves.append(node)


	def pushTotalError(self, W,L,Ws):
		self.setActivations(W,L)
		self.setPredictions(Ws)
		error = self.totalError()
		return error 

	def setActivations(self, W, L):
		for node in self.parentFirstOrderingLeaves: #set leaf activations first
			colno = node.word_id
			col = L[:,colno].copy()
			node.activation = numpy.tanh(col) #CHANGE 

		for node in self.parentFirstOrderingNonLeaves[::-1]: #do children before parents
			b = node.left.activation
			c = node.right.activation #the b and c per the paper
			bc = numpy.concatenate((b,c))
			node.activation = numpy.tanh(W*bc)

	def setPredictions(self, Ws):
		for node in self.parentFirstOrdering:
			a = node.activation
			y = softmax(Ws*a)
			node.y = y

	def totalError(self):
		total = 0
		for node in self.parentFirstOrdering:
			yk = node.y[node.score]
			error = -1*math.log(yk)
			total += error
		return total

	def setSoftmaxErrors(self, Ws):
		for node in self.parentFirstOrdering:
			error_vector = self.softmaxError(node, Ws)
			node.softmax_error = error_vector


	def getGradW(self):
		GradW = numpy.matlib.zeros((d, 2*d))
		for node in self.parentFirstOrderingNonLeaves: #only applies to nonleaves
			lhs = node.total_error
			b = node.left.activation
			c = node.right.activation
			bc = numpy.concatenate((b,c))
			temp = lhs*bc.T
			GradW = GradW + temp
		return GradW

	def getGradLSparse(self):
		index_grad = {}
		for node in self.parentFirstOrderingLeaves: 
			idx = node.word_id 
			grad = node.total_error
			if not idx in index_grad:
				index_grad[idx] = grad
			else:
				index_grad[idx]+=grad

		return index_grad


	def getGradWs(self):
		GradWs = numpy.matlib.zeros((NUM_CLASSES, d))
		for node in self.parentFirstOrdering:
			y = node.y
			t = node.t
			a = node.activation 
			temp = (y-t)*(a.T)
			GradWs = GradWs + temp
		return GradWs 

	def setTotalErrors(self, W):
		self.tree.parent_error = numpy.matlib.zeros((d,1))

		for node in self.parentFirstOrderingNonLeaves:
			node.total_error = node.parent_error+ node.softmax_error
			lhs = W.T*node.total_error
			b = node.left.activation
			c = node.right.activation
			bc = numpy.concatenate((b,c))
			fbc = numpy.tanh(bc)
			fbc2 = numpy.multiply(fbc, fbc)
			rhs = 1 - fbc2
			down_error = numpy.multiply(lhs, rhs)
			down_error_left = down_error[:d,0]
			down_error_right = down_error[d:,0]
			node.left.parent_error = down_error_left
			node.right.parent_error = down_error_right

		for node in self.parentFirstOrderingLeaves:
			node.total_error = node.parent_error + node.softmax_error




	def softmaxError(self, node, Ws):
		a = node.activation 
		y = node.y
		t = node.t 
		rhs = (1-numpy.multiply(a,a))
		lhs = Ws.T*(y-t)
		result = numpy.multiply(lhs, rhs)
		return result 

	'''
	Add all the nodes to self.parentFirstOrdering
	so the parent always comes before the children.
	'''
	def setOrdering(self, tree):
		if tree==None:
			return

		self.parentFirstOrdering.append(tree)
		self.setOrdering(tree.left)
		self.setOrdering(tree.right)
		

	'''
	Given a tree in string form, return the resulting tree.
	'''
	def buildTree(self, instring):
		curr = Node()
		firstspace = instring.find(" ")
		score = int(instring[1:firstspace])
		curr.setScore(score)

		isLeaf = (instring[firstspace+1]!="(")
		if isLeaf:
			word = instring[firstspace+1:-1]
			curr.makeLeaf(word)
			self.words.append(word)
			return curr

		split = self.getSplit(instring, firstspace) #return index of separating space
		first = instring[firstspace+1:split]
		second = instring[split+1:-1]

		curr.left = self.buildTree(first)
		curr.right = self.buildTree(second)

		curr.left.parent = curr
		curr.right.parent = curr

		return curr

	'''
	Given a string of the form 
	(number (paren_sequence_one)[space](parent_sequence_two))

	return the index of the [space] in between the paren sequences.
	fisrstspace gives the index right before the opening paren of 
	paren_sequence_one.
	'''
	def getSplit(self, instring, firstspace):
		currindex = firstspace + 2
		numparens = 1 #we start afterh the opening paren of paren_sequence_one
		while numparens > 0:
			if instring[currindex]=="(":
				numparens+=1
			elif instring[currindex]==")":
				numparens-=1
			currindex+=1
		return currindex
