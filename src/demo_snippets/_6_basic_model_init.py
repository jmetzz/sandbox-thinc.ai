from thinc.api import chain, Relu, Softmax
 
n_hidden = 32
dropout = 0.2

model = chain(
    Relu(nO=n_hidden, dropout=dropout), 
    Relu(nO=n_hidden, dropout=dropout), 
    Softmax()
)


import numpy
from thinc.api import Linear, glorot_uniform_init

model = Linear(nI=16, nO=10, init_W=glorot_uniform_init)


X = numpy.zeros((128, 16), dtype="f")
Y = numpy.zeros((128, 10), dtype="f")
model.initialize(X=X, Y=Y)


