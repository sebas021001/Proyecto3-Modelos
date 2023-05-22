"""
Ejemplo de script con docstrings.

Tiene errores intencionales de PEP-8 y de PEP-257.
"""

import numpy as np

tau=2*np.pi

def circ(r):
    """Calcula el perímetro de una circunferencia de radio r.
    Según la fórmula antiquísima 2*pi*r. La circunferencia es una curva plana y cerrada tal que todos sus puntos están a igual distancia del centro (Wikipedia).
    Parameters
    ----------
    r : float
        El radio del círculo.

    Returns
    -------
    float
        El perímetro de la circunferencia.
    """
    return f'El perímetro de la circunferencia es {tau*r:.2f}\n'

print(circ(3))
print('Información sobre la función:\n')
print(f'-- {circ.__name__} --')
print(circ.__doc__)