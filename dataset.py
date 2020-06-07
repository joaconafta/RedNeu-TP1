import pandas as pd

# normalización de valores entre -1 y 1
def normalize(data):
  for c in data.columns:
    M = data[c].max()
    m = data[c].min()
    data[c] = (data[c] - m) / (M - m)
    data[c] = data[c] * 2 - 1

# carga de datos dadas las columnas usadas como entrada y target,
# y una funcion aplicada al target
def load_dataset(ds_file, input_break, target_break, apply_target):
  df = pd.read_csv(ds_file, header=None)

  # datos de entrada
  data_input = pd.DataFrame(df[df.columns[
    input_break[0]:input_break[1]]
  ])

  normalize(data_input)

  # target
  data_target = df[df.columns[
    target_break[0]:target_break[1]]
  ]

  input = data_input.to_numpy()
  target = apply_target(data_target)

  return (input, target)
