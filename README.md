# Redes neuronales artificiales - Trabajo práctico 1

Implementación de modelos de _perceptrón simple_ y _perceptrón multicapa_.

## Ejercicio 1

Problema de clasificación: predecir el diagnóstico final de cáncer de mamas.

Dataset: resultados de un examen especifico que es utilizado en el diagnostico de
cáncer de mamas.

Cada entrada correspondiente a los datos obtenidos para distintos pacientes y
contiene 10 características provenientes de imágenes digitalizadas de muestras de
células, y el diagnostico final en donde se indica si la muestra analizada
pertenecía a un tumor maligno o benigno.

### Correr

```
python ej1.py MODELO DATASET
```

Para entrenar usar:

- `MODELO` path donde se guarda el modelo. Este archivo no debe existir.
- `DATASET` path al dataset de entrenamiento

Para testar usar:

- `MODELO` path al modelo entrenado. Este archivo debe existir.
- `DATASET` path al dataset de testing

## Ejercicio 2

Problema de regresión: predecir los valores de carga energética para la calefacción y
refrigeración de edificios.

Dataset: análisis energético de edificios de distintas formas que difieren con
respecto a la superficie y distribución de las áreas de reflejo, la orientación y
otros parámetros.

Cada entrada en el conjunto de datos corresponde a las características de un edificio
distinto junto a dos valores reales que representan la cantidad de energía necesaria
para realizar una calefacción y refrigeración adecuadas.

## Setup

Python: v3.6.5

Para instalar version de Python

1. Instalar `pyenv` - https://github.com/pyenv/pyenv#installation

  En mac:

  ```
  brew update
  brew install pyenv
  ```

2. Instalar v3.6.5

  ```
  pyenv install 3.6.5
  ```

3. Activar v3.6.5 en terminal

  ```
  eval "$(pyenv init -)"
  pyenv global 3.6.5
  ```

4. Verificar version

  ```
  python --version
  > Python 3.6.5
  ```

5. Instalar dependencias

  ```
  pip install -r requirements.txt
  ```
