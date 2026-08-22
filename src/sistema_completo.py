"""
sistema_completo.py
--------------------
Fase 5: Integración y Sistema Completo.

Une las cuatro piezas del proyecto:
  1. Preprocesamiento NLP           (procesador_nlp.ProcesadorNLP)
  2. Clasificación supervisada      (clasificador.ClasificadorTextos)
  3. Clustering no supervisado      (clusterizador.ClusterizadorTextos)
  4. Generación con cadenas Markov  (generador_markov.GeneradorMarkov)
"""

import pandas as pd

from clasificador import ClasificadorTextos
from clusterizador import ClusterizadorTextos
from generador_markov import GeneradorMarkov
from procesador_nlp import ProcesadorNLP


class SistemaNLPCompleto:
    def __init__(self, idioma="spanish"):
        self.procesador = ProcesadorNLP(idioma)
        self.clasificador = None
        self.clusterizador = None
        self.generadores = {}
        self.textos = None
        self.etiquetas = None

    def cargar_corpus(self, ruta_base):
        self.textos, self.etiquetas = self.procesador.cargar_corpus(ruta_base)
        return self.textos, self.etiquetas

    def entrenar_clasificadores(self, verbose=True):
        self.clasificador = ClasificadorTextos(self.textos, self.etiquetas)
        self.clasificador.dividir_datos()
        self.clasificador.entrenar_clasificadores()
        return self.clasificador.evaluar_modelos(verbose=verbose)

    def hacer_clustering(self, n_clusters=3):
        self.clusterizador = ClusterizadorTextos(self.textos)
        self.clusterizador.preparar_datos()
        self.clusterizador.clusterizar(n_clusters)
        return self.clusterizador.analizar_clusters(self.etiquetas)

    def entrenar_generadores(self):
        self.generadores = {}
        for categoria in sorted(set(self.etiquetas)):
            textos_categoria = [t for t, e in zip(self.textos, self.etiquetas) if e == categoria]
            modelo1 = GeneradorMarkov(orden=1)
            modelo2 = GeneradorMarkov(orden=2)
            modelo1.entrenar_multiples(textos_categoria)
            modelo2.entrenar_multiples(textos_categoria)
            self.generadores[categoria] = {"orden1": modelo1, "orden2": modelo2}
        print(f"Generadores entrenados para {len(self.generadores)} categorías")
        return self.generadores

    def predecir_y_generar(self, prompt, num_palabras=40, modelo_clasificacion="Naive Bayes",
                            orden_generador="orden2"):
        """Predice la categoría de sentimiento de un texto y genera una
        continuación de estilo similar usando el modelo Markov de esa categoría."""
        if self.clasificador is None:
            raise ValueError("Primero debes entrenar clasificadores")

        X_prompt = self.clasificador.vectorizer.transform([prompt])
        categoria = self.clasificador.modelos[modelo_clasificacion].predict(X_prompt)[0]

        if categoria not in self.generadores:
            return {"categoria_predicha": categoria, "texto_generado": f"Sin generador para {categoria}",
                    "prompt_original": prompt}

        modelo = self.generadores[categoria][orden_generador]
        texto_generado = modelo.generar(num_palabras, prompt)

        return {
            "categoria_predicha": categoria,
            "texto_generado": texto_generado,
            "prompt_original": prompt,
        }

    def generar_reporte(self, relatos_sin_etiqueta):
        """Genera un reporte completo (clasificación + cluster + generación)
        para una lista de textos que no tienen etiqueta de sentimiento."""
        analisis_clusters = self.clusterizador.analizar_clusters(self.etiquetas)
        resultados = []

        for i, texto in enumerate(relatos_sin_etiqueta):
            X_texto = self.clasificador.vectorizer.transform([texto])
            pred_nb = self.clasificador.modelos["Naive Bayes"].predict(X_texto)[0]
            pred_lr = self.clasificador.modelos["Logistic Regression"].predict(X_texto)[0]

            X_cluster = self.clusterizador.vectorizer.transform([texto])
            cluster = int(self.clusterizador.kmeans.predict(X_cluster)[0])
            categoria_cluster = analisis_clusters[cluster]["categoria_principal"]

            consenso = pred_nb if pred_nb == pred_lr else f"{pred_nb} / {pred_lr} (sin consenso)"

            generacion = self.predecir_y_generar(" ".join(texto.split()[:5]))

            resultados.append(
                {
                    "texto_id": i + 1,
                    "texto": texto,
                    "clasificacion_nb": pred_nb,
                    "clasificacion_lr": pred_lr,
                    "consenso": consenso,
                    "cluster": cluster,
                    "categoria_cluster": categoria_cluster,
                    "generacion_estilo": generacion["categoria_predicha"],
                    "texto_generado": generacion["texto_generado"],
                }
            )

        return pd.DataFrame(resultados)
