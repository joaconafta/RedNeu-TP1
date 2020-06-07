import matplotlib.pyplot as plt

# grafica errores en funcion de epocas
def plot_error(errors):
  plt.xlabel('epoch')
  plt.title('error')
  plt.title('entrenamiento')

  plt.plot(errors)
  plt.show()
