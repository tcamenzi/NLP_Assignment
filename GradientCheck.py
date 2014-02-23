
'''
This gradient check code was used during developing / debugging, but is no longer being used. 
It performs gradient check for a single training instance (here named te)
''' 


a = "(3 (2 The) (2 Rock))" #trivial training case for debugging.
te = training_instances[4]  # TrainingInstance(a)
for node in te.parentFirstOrderingLeaves:
	node.addID(word_index)

# id1 = word_index["The"]
# id2 = word_index["Rock"]
# L[:,id1] = numpy.matrix('1;0')
# L[:,id2] = numpy.matrix('0;1')
# W = numpy.matrix('1 0 2 5; 3 0 4 6').astype(float)
te.setActivations(W,L)
# Ws = numpy.matrix('1 0; 1 0; 1 0; 0 2; 0 2').astype(float)
te.setPredictions(Ws)
error = te.totalError()
print "error: ", error
print te.tree.activation

def gradCheckWs(te, W, L, Ws):
	error = te.pushTotalError(W,L,Ws) #this sets y,t,a and gives the error
	WsGrad = te.getGradWs()

	eps = .0001
	WsGradApprox = numpy.matlib.zeros((config.NUM_CLASSES, d))
	Ws_up = Ws.copy()
	Ws_down = Ws.copy()
	 
	for i in range(config.NUM_CLASSES):
		for j in range(config.d):
			Ws_up[i,j]+=eps
			Ws_down[i,j]-=eps

			error1 = te.pushTotalError(W,L,Ws_up)
			error2 = te.pushTotalError(W,L,Ws_down)
			result = (error1-error2)/(2*eps)
			WsGradApprox[i,j] = result

			Ws_up[i,j]-=eps
			Ws_down[i,j]+=eps

	return WsGrad, WsGradApprox

def gradCheckW(te, W, L, Ws):
	error = te.pushTotalError(W,L,Ws) #this sets y,t,a and gives the error
	WGrad = te.getGradW()

	eps = .0001
	WGradApprox = numpy.matlib.zeros((config.d, 2*config.d))
	W_up = W.copy()
	W_down = W.copy()
	
	for i in range(config.d):
		for j in range(2*d):
			W_up[i,j]+=eps
			W_down[i,j]-=eps

			error1 = te.pushTotalError(W_up,L,Ws)
			error2 = te.pushTotalError(W_down,L,Ws)
			result = (error1-error2)/(2*eps)
			WGradApprox[i,j] = result

			W_up[i,j]-=eps
			W_down[i,j]+=eps

	return WGrad, WGradApprox

def gradCheckLSparse(te, W, L, Ws):
	error = te.pushTotalError(W,L,Ws) #this sets y,t,a and gives the error
	LGradSparse = te.getGradLSparse(Ws)

	eps = .0001
	LGradSparseApprox = {}
	L_up = L.copy()
	L_down = L.copy()
	
	for j in LGradSparse:
		LGradSparseApprox[j] = numpy.matlib.zeros((config.d,1))

	for i in range(config.d):
		for j in LGradSparse:
			L_up[i,j]+=eps
			L_down[i,j]-=eps


			error1 = te.pushTotalError(W,L_up,Ws)
			error2 = te.pushTotalError(W,L_down,Ws)
			result = (error1-error2)/(2*eps)
			LGradSparseApprox[j][i,0] = result

			L_up[i,j]-=eps
			L_down[i,j]+=eps

	return LGradSparse, LGradSparseApprox

PRINT_GRADCHECK = False 
if PRINT_GRADCHECK:
	WsGrad, WsGradApprox = gradCheckWs(te, W,L,Ws)
	print "WsGrad: ", WsGrad
	print "WsGradApprox: ", WsGradApprox
	print "Difference: ", WsGrad - WsGradApprox

	te.setSoftmaxErrors(Ws)
	te.setTotalErrors(W)

	WGrad, WGradApprox = gradCheckW(te, W,L,Ws)
	print "WGrad: ", WGrad
	print "WGradApprox: ", WGradApprox
	print "Difference: ", WGrad - WGradApprox

	gradLSparse, gradLSparseApprox = gradCheckLSparse(te, W, L, Ws)
	for i in gradLSparse:
		print "LGrad, LGradApprox for word %s:" % index_word[i]
		print gradLSparse[i].T
		print gradLSparseApprox[i].T
		print (gradLSparse[i]-gradLSparseApprox[i]).T
'''
