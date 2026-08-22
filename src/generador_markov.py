"""
generador_markov.py
--------------------
Fase 4: Modelos Generativos (cadenas de Markov de orden 1 y 2).
"""

import random
from collections import defaultdict


class GeneradorMarkov:
    def __init__(self, orden=1):
        """orden: 1 (primer orden) o 2 (segundo orden)."""
        self.orden = orden
        self.cadena = defaultdict(list)
        self.inicios = []
        self.textos_entrenados = []

    def entrenar(self, texto):
        """Entrena el modelo con un texto."""
        self.textos_entrenados.append(texto)
        tokens = texto.split()

        if len(tokens) < self.orden + 1:
            return

        self.inicios.append(tuple(tokens[: self.orden]))

        for i in range(len(tokens) - self.orden):
            estado = tuple(tokens[i:i + self.orden])
            siguiente = tokens[i + self.orden]
            self.cadena[estado].append(siguiente)

    def entrenar_multiples(self, textos):
        """Entrena con múltiples textos."""
        for texto in textos:
            self.entrenar(texto)
        print(f"Entrenado con {len(textos)} textos (orden {self.orden})")
        print(f"Estados distintos: {len(self.cadena)}")

    def _estado_inicial(self, semilla=None):
        if semilla is None:
            if not self.inicios:
                return None
            return random.choice(self.inicios)

        semilla_tokens = semilla.split()
        if len(semilla_tokens) >= self.orden:
            return tuple(semilla_tokens[-self.orden:])

        # semilla más corta que el orden: completar con un inicio real
        estado = list(semilla_tokens)
        relleno = random.choice(self.inicios) if self.inicios else ("...",) * self.orden
        i = 0
        while len(estado) < self.orden:
            estado.append(relleno[i % len(relleno)])
            i += 1
        return tuple(estado)

    def generar(self, num_palabras=50, semilla=None):
        """Genera texto a partir de una semilla opcional."""
        estado = self._estado_inicial(semilla)
        if estado is None:
            return "No hay datos para generar"

        resultado = list(estado)

        for _ in range(max(0, num_palabras - self.orden)):
            if estado not in self.cadena:
                if not self.cadena:
                    break
                estado = random.choice(list(self.cadena.keys()))
                resultado.extend(estado)
                continue

            siguiente = random.choice(self.cadena[estado])
            resultado.append(siguiente)

            if self.orden == 1:
                estado = (siguiente,)
            else:
                estado = tuple(list(estado[1:]) + [siguiente])

        return " ".join(resultado[:num_palabras])
