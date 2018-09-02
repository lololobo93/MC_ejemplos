Ejemplos de simulaciones Monte Carlo para diferentes sistemas.

- Esferas duras en una dimensión.
- Esferas duras bidimensional.
- Modelo de Ising bidimensional por medio del algoritmo de Metrópolis.
- Modelo de Ising bidimensional por medio del algoritmo de Wolf.
- Modelo de q-Potts por medio del algoritmo de Metrópolis con selección por Heat-Bath.

Se observa que el algoritmo de Wolf es más rápido y presenta mejores resultados en comparación con el algoritmo de Metrópolis, lo cual es consistente con la teoría presentada en [Monte Carlo Methods in Statistical Physics](http://itf.fys.kuleuven.be/~fpspXIII/material/Barkema_FPSPXIII.pdf) de Newman y Barkema. Finalmente, se estudia el modelo de q-Potts resuelto por medio del algoritmo de Metrópolis con selección por Heat-Bath, este modelo también se puede abordar usando el algoritmo de Wolf, el cual es un buen ejercicio adicional, pero este no será incluido en estos programas.

Los parámetros que coloco para el número de pasos Monte Carlo se pueden modificar para hacer el código más rápido o preciso. Los valores que coloco solo son demostrativos y en mi computadora personal permiten llevar los cálculos de forma rápida.