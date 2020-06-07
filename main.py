from dataset import load_dataset
from os import path
import numpy as np
import pickle

B = 1 # para cambiar buckets

def init_w(S):
  return [
    0 if i == 0 else
      np.random.normal(
      0, S[0]**(-1/2), (S[i-1]+1, S[i])
    )
    for i in range(len(S))
  ]

def init_dw(W):
  return [
    0 if i == 0 else
      np.zeros_like(W[i])
    for i in range(len(W))
  ]

def init_y(S):
  _Y = []
  L = len(S)

  _Y = [np.zeros((B, S[i]+1)) for i in range(L - 1)]

  _Y.append(np.zeros((B, S[L-1])))

  return _Y

# agregar y sacar umbrales.
def bias_add(V):
  bias = -np.ones((len(V), 1))
  return np.concatenate((V, bias), axis=1)

def bias_sub(V):
  return V[:,:-1]

def activation(S, Xh, W):
  Y = init_y(S)
  _Y = Xh

  L = len(S)

  for k in range(1, L):
    Y[k-1][:] = bias_add(_Y)
    _Y = np.tanh(np.dot(Y[k-1], W[k])) # aca setea funcion g

  Y[L-1][:] = _Y

  return Y

def correction(S, Zh, W, Y, lr):
  L = len(S)

  dW = init_dw(W) # Inicializo vector de matrices dW con dimensiones de W.
  D = init_y(S) # Inicializo vector D con dimensiones de Y.
  E = Zh - Y[L-1] # Error en la salida.
  dY = 1 - np.square(Y[L-1]) # Derivada de la función de activación en la salida.
  D[L-1] = E*dY

  for k in range(L-1, 0, -1):
    dW[k] = lr * np.dot(Y[k-1].T, D[k])
    E = np.dot(D[k], W[k].T)
    dY = 1 - np.square(Y[k-1])
    D[k-1] = bias_sub(E*dY)

  return dW

def adaptation(W, dW):
  for k in range(1, len(W)):
    W[k] += dW[k]
  return W

def estimation(Zh, Y):
  return np.mean(np.sum(np.square(Zh-Y[-1]), axis=1))

def train(X, Z, S, max_epoch, lr):
  W = init_w(S)

  errors = []
  error = 1
  t = 0

  while(error > 0.001 and t < max_epoch):
    if t % 100 == 0:
      print('epoch {} - error {}/{}'.format(t, error, len(X)))

    error = 0
    H = np.random.permutation(len(X)) # mini-lotes (recorro instancias en ordenes random)

    for h in H:
      # TODO: if h+B <= P: - para B > 1
      Y = activation(S, X[h:h + B], W) # pasamos una vez cada instancia
      dW = correction(S, Z[h:h+B], W, Y, lr) # calculamos dW
      W = adaptation(W, dW) # actualizamos W con dW
      error += estimation(Z[h:h+B], Y) # suma de los errores de cada instancia

    t += 1
    errors.append(error) # errores x epoch

  return errors, W

def test(X, Z, S, W):
  Y = [np.sign(activation(S, X[i:i+1], W)[-1][0]) for i in range(len(X))]
  return np.mean(np.where(Y == Z, 1, 0))

def main(
  argv, input_break, target_break, apply_target,
  S, max_epoch, lr
):
  if (len(argv) != 3):
    print('Error en la entrada')

  [_, model, data] = argv

  (input, target) = load_dataset(
    data,
    input_break,
    target_break,
    apply_target
  )

  if (not path.exists(model + '.p')):
    errors, W = train(input, target, S, max_epoch, lr)
    pickle.dump(W, open(model + '.p', 'wb'))
  else:
    W = pickle.load(open(model + '.p', 'rb'))
    r = test(input, target, S, W)
    print('precisión: {}'.format(r))
