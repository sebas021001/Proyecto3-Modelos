"""Este codigo contiene funciones para el analisis del consumo electrico.

Se encarga de tomar datos proveidos por un archivo json y convertirlo
a dataFrame para de esta manera analizar y visualizar ciertos datos por
medio de la implementación de funciones. Entre las funciones se encunetra,
la busqueda de datos de consumo a una hora particular, la busqueda del modelo
que mejor se ajusta a los datos de consumo a una hora particular, creación de
histogramas tanto en 2d como en 3d.

Autor: Sebastián Vasquez Hidalgo
Fecha: 25-05-2023
"""

import pandas as pd
import random
import json
from fitter import Fitter
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


def asignacion_horas(digitos):
    """Elige dos horas con datos de carne.

    Elige una hora A en periodo punta
    y una hora B de los otros periodos,
    con los dígitos del carné como "seed".

    Parameters
    ----------
    digitos : integer
              Digitos del carné

    Returns
    -------
    hora_A : integer
             hora en periodo punta
    hora_B : integer
             hora en los otros periodos
    """
    random.seed(digitos)
    punta = [11, 12, 18, 19, 20]
    valle = [7, 8, 9, 10, 13, 14, 15, 16, 17]
    nocturno = [21, 22, 23, 0, 1, 2, 3, 4, 5, 6]
    otro = valle + nocturno
    hora_A = random.choice(punta)
    hora_B = random.choice(otro)
    return hora_A, hora_B


horas = asignacion_horas(8310)
print(f'Las horas asignadas son {horas[0]} y {horas[1]}.')

with open("demandaMW_2019.json", "r") as file:
    data = json.load(file)

df = pd.DataFrame(data["data"])

# Se cambia el formato de fechaHora para poder buscar la hora especifica
# deseada
df['fechaHora'] = pd.to_datetime(df['fechaHora'],
                                 format='%Y-%m-%d %H:%M:%S.%f')


def datos_hora(hora):
    """Encuentra informacion de consumo para la hora deseada.

    Se encarga de buscar en la columna 'fechaHora' la hora correspondiente
    a la primer hora asignada por la función 'asignacion_horas' y extrar
    la información de consumo en MW para esa hora especifica.

    Parameters
    ----------
    hora : integer
           Hora para la cual se desean los datos de consumo.

    Returns
    -------
    series
         Datos de consumo en MW para la hora deseada.

    """
    data_hour = df[df['fechaHora'].dt.hour == hora]['MW']

    return data_hour


data_hourA = datos_hora(horas[0])

print()
print(f'Los datos de consumo en MW para la hora {horas[0]} %s' %
      f'son los siguientes:\n', data_hourA)


def modelo_hora(hora):
    """Busca el mejor modelo para el consumo de una hora deseada.

    Se encarga de obtener el modelo que mejor se ajusta a los datos
    obtenidos para la primer hora asignada.

    Parameters
    ----------
    hora : integer
           Hora para la cual se obtiene los datos de consumo.

    Returns
    -------
    string
         Modelo que mejor se ajusta a los datos.
    """
    data_hour = df[df['fechaHora'].dt.hour == hora]['MW']
    fitted = Fitter(data_hour)
    fitted.fit()
    best_model = fitted.get_best(method='sumsquare_error')
    best_curve = next(iter(best_model))

    return best_curve


mejor_curva = modelo_hora(horas[0])
print(f'El modelo que mejor se ajusta a los datos de consumo para la hora %s' %
      f'{horas[0]} es {mejor_curva}')

print()
print(f'El histogram del consumo para las {horas[0]} horas se muestra %s' %
      f'se muestra en la figura 1')


def visualizacion_hora(hora, curva):
    """Grafica curva del modelo de mejor ajuste sobre hisrograma.

    Se encarga de graficar la curva del modelo que mejor se ajusta
    a los datos de consumo de la hora deseada sobre el histograma
    de estos datos.

    Parameters
    ----------
    hora : integer
           Hora para la cual se obtienen los datos de consumo.
    curva : string
            Modelo que mejor se ajusta a los datos de consumo.

    Returns
    -------
    plot
       Gráfico de la curva del modelo de mejor ajuste sobre histograma.
    """
    data_hour = df[df['fechaHora'].dt.hour == hora]['MW']
    params = getattr(stats, curva).fit(data_hour)
    ran = np.linspace(data_hour.min(), data_hour.max(), 100)

    plt.figure()
    plt.hist(
        data_hour,
        density=True,
        bins=100,
        align="mid",
        color="orange",
        ec="k",
        label="MW",
    )

    plt.plot(
        ran,
        getattr(stats, curva).pdf(ran, *params),
        linewidth=2,
        label=f"Model ({curva})",
    )

    plt.xlabel('MW')
    plt.ylabel('Frecuencia')
    plt.title(f"Figura 1: Distribuciñon de MW a las {hora} horas")
    plt.legend()


visualizacion_hora(horas[0], mejor_curva)


def correlacion_horas(hora1, hora2):
    """Correlación entre las horas A y B.

    Determina la correlacion entre las horas asignadas por las función
    'asignacion_horas'.

    Parameters
    ----------
    hora1 : integer
            Primer hora asignada.
    hora2 : integer
            Segunda hora asignada.

    Returns
    -------
    float
        Valor de la correlación entre las horas 1 y 2.
    """
    MW_hour1 = df[df['fechaHora'].dt.hour == hora1]['MW']
    MW_hour2 = df[df['fechaHora'].dt.hour == hora2]['MW']

    correlation = np.corrcoef(MW_hour1, MW_hour2)[0, 1]

    return correlation


correlacion = correlacion_horas(horas[0], horas[1])
print(f'El conficiente de correlación entre las horas %s' %
      f'{horas[0]} y {horas[1]} es de {correlacion:.4}')

print()
print(f'El histograma 3D del consumo en las horas %s' %
      f'{horas[0]} y {horas[1]} se mestra en la figura 2.')


def visualizacion_horas(hora1, hora2):
    """Crea histograma 3D del consumo en MW de las horas asignadas.

    Se encarga de crear un histograma en 3D en donde se muestra el consumo
    en MW de la hora1 y la hora 2.

    Parameters
    ----------
    hora1 : integer
            Primer hora asignada.
    hora2 : integer
            Segunda hora asignada.

    Returns
    -------
    plot
        Histograma en  3D.
    """
    MW_hour1 = df[df['fechaHora'].dt.hour == hora1]['MW']
    MW_hour2 = df[df['fechaHora'].dt.hour == hora2]['MW']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    hist, xedges, yedges = np.histogram2d(MW_hour1, MW_hour2, bins=50)
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    dx = dy = 0.8 * (xedges[1] - xedges[0])
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='blue', edgecolor='black')

    ax.set_xlabel(f'Consumo a las {hora1} horas.')
    ax.set_ylabel(f'Consumo a las {hora2} horas.')
    ax.set_zlabel('Frecuencia')
    ax.set_title(f'Figura 2: Histograma 3D para las horas {hora1} y {hora2}')


visualizacion_horas(horas[0], horas[1])
plt.show()
