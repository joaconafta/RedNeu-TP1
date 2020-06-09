from matplotlib import pyplot as plt,cm
import numpy as np

# grafica errores en funcion de epocas
def plot_error(errors):
  plt.xlabel('epoch')
  plt.title('error')
  plt.title('entrenamiento')

  plt.plot(errors)
  plt.show()

def plot_target(Y,Z):
  #Calefaccion
  #plt.title("Calefacción")
  #plt.scatter(Y.index, Y[0], label='Activación',s=70)
  #plt.scatter(Y.index, Z[0], label='Esperado', s=25)
  #plt.show()
  fig, ax1 = plt.subplots()
  plt.title("Calefacción")
  ax1.set_xlabel('instancias')
  ax1.set_ylabel('valor de los parámetros')
  ax1.scatter(Y.index, Y[0], label='Activación', s=70)
  ax1.scatter(Y.index, Z[0], label='Esperado', color='orange', s=25)
  fig.legend()
  plt.show()

  #Refrigeracion
  #plt.title("Refrigeración")
  #plt.scatter(Y.index, Y[1], label='Activación', s=70)
  #plt.scatter(Y.index, Z[1], label='Esperado', s=25)
  #plt.show()
  fig, ax1 = plt.subplots()
  plt.title("Refrigeración")
  ax1.set_xlabel('instancias')
  ax1.set_ylabel('valor de los parámetros')
  ax1.scatter(Y.index, Y[1], label='Activación', s=70)
  ax1.scatter(Y.index, Z[1], label='Esperado', color='orange', s=25)
  fig.legend()
  plt.show()

def plot_target_1(Y,Z):
  Y2 = np.sign(Y)
  Y2 = (Y2 + 1)/2

  Y = (Y + 1) /2
  Z = (Z + 1) /2

  M = np.concatenate((Y, Y2,Z), axis=1)
  M = M[:30]

  fig, ax = plt.subplots(figsize=(5, 3))
  fig.subplots_adjust(bottom=0.15, left=0.2)
  ax.matshow( M.T, cmap=cm.gray)
  ax.set_xlabel('instancias')
  ax.set_yticklabels(['a','Act','sign(Act)','Esp'])
  plt.show()

def plot_scat_1(Y,Z):
  fig, ax1 = plt.subplots()
  plt.title("")
  ax1.set_xlabel('instancias')
  ax1.set_ylabel('clasificación')
  ax1.scatter(range(len(Y)), Y, label='Activación', s=70)
  ax1.scatter(range(len(Y)), Z, label='Esperado', color='orange', s=25)
  fig.legend()
  plt.show()