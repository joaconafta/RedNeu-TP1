from dataset import load_dataset
from os import path
import numpy as np
import pickle
from plotter import plot_error
from plotter import plot_target
from plotter import plot_target_1
import pandas as pd

# funciones de inicialización
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

def init_y(S, B):
  _Y = []
  L = len(S)

  _Y = [np.zeros((B, S[i]+1)) for i in range(L - 1)]

  _Y.append(np.zeros((B, S[L-1])))

  return _Y

# agregar y sacar umbrales
def bias_add(V):
  bias = -np.ones((len(V), 1))
  return np.concatenate((V, bias), axis=1)

def bias_sub(V):
  return V[:,:-1]

# funciones de entrenamiento

# activacion feed forward
def activation(S, Xh, W, B):
  Y = init_y(S, B)
  _Y = Xh

  L = len(S)

  for k in range(1, L):
    Y[k-1][:] = bias_add(_Y)
    _Y = np.tanh(np.dot(Y[k-1], W[k])) # función de activación g

  Y[L-1][:] = _Y

  return Y

# correccion backpropagation
def correction(S, Zh, W, Y, lr, B):
  L = len(S)

  dW = init_dw(W)
  D = init_y(S, B)
  E = Zh - Y[L-1]
  dY = 1 - np.square(Y[L-1]) # derivada de la función de activación
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

def train(X, Z, S, max_epoch, lr, B):
  # inicialización de matriz de pesos
  W = init_w(S)

  errors = []
  error = 1
  t = 0

  print('S: {}'.format(S))
  print('Máxima #epocas: {}'.format(max_epoch))
  print('LR: {}'.format(lr))
  print('B (mini-lotes): {}'.format(B))

  while(error > 0.001 and t < max_epoch):
    # resultado parcial
    if t % 100 == 0 and t > 0:
      print('epoch: {}; error: {}'.format(t, error))

    error = 0

    # mini-lotes - mezcla de orden de instancias
    stochastic = np.random.permutation(len(X))

    for h in range(0, len(X) - B, B):
      h = stochastic[h:h+B]
      Y = activation(S, X[h], W, B)
      dW = correction(S, Z[h], W, Y, lr, B)
      W = adaptation(W, dW)
      error += estimation(Z[h], Y)

    t += 1
    errors.append(error)

  return errors, W

def test(X, Z, S, W):
  Y = [(activation(S, X[i:i+1], W, 1)[-1][0])
    for i in range(len(X))]
  
  Y = pd.DataFrame(Y)
  Z = pd.DataFrame(Z)

  if Z.shape[1] ==  1:
    #ej1
    acc = np.mean(np.where(np.sign(Y) == Z, 1, 0)) # promedio de aciertos
  else:
    #ej2
    plot_target(Y, Z)
    acc = np.mean(np.sum(np.square(Z-Y), axis=0)) # error cuadratico medio

  return  acc, Y, Z


def main(
  argv, input_break, target_break, apply_target,
  S, max_epoch, lr, B = 1
):
  if (len(argv) != 3):
    print('Error en la entrada')

  [_, model, data] = argv

  # lectura y pre procesamiento de datos
  (input, target) = load_dataset(
    data,
    input_break,
    target_break,
    apply_target
  )

  if (not path.exists(model + '.p')):
    # entrenamiento
    errors, W = train(input, target, S, max_epoch, lr, B)

    plot_error(errors)
    pickle.dump(errors, open(model + '_errors.p', 'wb'))
    pickle.dump(W, open(model + '.p', 'wb'))
  else:
    # testing

    # carga de modelo entrenado
    W = pickle.load(open(model + '.p', 'rb'))

    # testeo
    r,Y,Z = test(input, target, S, W)

    if Z.shape[1] ==  1:
      print('precisión: {}  (aciertos/total)'.format(r))
    else:
      print('error cuadratico medio: {}'.format(r))


class Model:
  def __init__(
    self,
    data,
    input_break,
    target_break,
    apply_target
  ):
    (_input, _target) = load_dataset(
      data,
      input_break,
      target_break,
      apply_target
    )

    self.input = _input
    self.target = _target

  def train(self, S, max_epoch, lr, train_break, B):
    self.S = S
    self.max_epoch = max_epoch
    self.lr = lr

    errors, W = train(
      self.input[train_break[0]:train_break[1]],
      self.target[train_break[0]:train_break[1]],
      self.S, self.max_epoch, self.lr, B
    )

    self.model = W
    return errors

  def test(self, test_break):
    return test(
      self.input[test_break[0]:test_break[1]],
      self.target[test_break[0]:test_break[1]],
      self.S, self.model
    )

  '''def test_scatter(self, test_break):
    X = self.input[test_break[0]:test_break[1]]
    Y = [(activation(self.S, X[i:i+1], self.model, 1)[-1][0])
      for i in range(len(X))]

    Z = self.target[test_break[0]:test_break[1]]

    Y = pd.DataFrame(Y)
    Z = pd.DataFrame(Z)

    plot_target(Y, Z)

    return np.mean(np.sum(np.square(Z-Y), axis=0))'''

  def exp(self, S, max_epoch, lr, train_break, test_break, B):
    errors = self.train(S, max_epoch, lr, train_break, B)
    prec, Y, Z = self.test(test_break)
    plot_target_1(Y, Z)
    plot_error(errors)
    print('precision: {}'.format(prec))

    return errors

  def exp_2(self, S, max_epoch, lr, train_break, test_break, B):
    errors = self.train(S, max_epoch, lr, train_break, B)
    ecm, Y, Z = self.test(test_break)

    print('error cuadratico medio: {}'.format(ecm))
    return errors