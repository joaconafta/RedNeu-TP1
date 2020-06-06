import sys

def main():
  if (len(sys.argv) != 3):
    print('Error en la entrada')

  [_, model, data] = sys.argv

  
  print(model)
  print(data)


