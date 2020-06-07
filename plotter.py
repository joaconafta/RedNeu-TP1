import matplotlib.pyplot as plt

def plot_error(errors):
  plt.xlabel('epoch')
  plt.title('error')
  plt.title('entrenamiento')

  plt.plot(errors)
  plt.show()
