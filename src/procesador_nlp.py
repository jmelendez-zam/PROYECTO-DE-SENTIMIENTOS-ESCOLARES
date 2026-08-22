"""
procesador_nlp.py
------------------
Fase 1: Preprocesamiento y NLP.

Implementa la clase ProcesadorNLP siguiendo la guía de la tarea
(tarea_proyecto.md, Fase 1), adaptada a la estructura de carpetas de
este proyecto: data/corpus_etiquetado/<categoria>/*.txt
"""

import os
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


def _asegurar_recursos_nltk():
    for recurso in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.data.find(
                f"tokenizers/{recurso}" if "punkt" in recurso else f"corpora/{recurso}"
            )
        except LookupError:
            nltk.download(recurso, quiet=True)


class ProcesadorNLP:
    def __init__(self, idioma="spanish"):
        """Inicializa el procesador con el idioma adecuado."""
        _asegurar_recursos_nltk()
        self.idioma = idioma
        self.stemmer = SnowballStemmer(idioma)
        self.stopwords = set(stopwords.words(idioma))
        self.textos = []
        self.etiquetas = []
        self.ids = []

    def cargar_corpus(self, ruta_base):
        """
        Carga todos los textos del corpus.

        ESTRUCTURA ESPERADA:
        ruta_base/
            categoria1/
                texto1.txt
            categoria2/
                texto2.txt
            ...
        """
        self.textos, self.etiquetas, self.ids = [], [], []

        for categoria in sorted(os.listdir(ruta_base)):
            ruta_categoria = os.path.join(ruta_base, categoria)
            if os.path.isdir(ruta_categoria):
                for archivo in sorted(os.listdir(ruta_categoria)):
                    if archivo.endswith(".txt"):
                        ruta_archivo = os.path.join(ruta_categoria, archivo)
                        with open(ruta_archivo, "r", encoding="utf-8") as f:
                            self.textos.append(f.read())
                        self.etiquetas.append(categoria)
                        self.ids.append(archivo.replace(".txt", ""))

        print(f"Corpus cargado: {len(self.textos)} textos")
        print(f"Categorías: {sorted(set(self.etiquetas))}")
        return self.textos, self.etiquetas

    def limpiar_texto(self, texto):
        """Minúsculas, sin signos de puntuación ni números, espacios normalizados.
        Se conservan tildes y la ñ, propias del español."""
        texto = texto.lower()
        texto = re.sub(r"[^\w\sáéíóúüñ]", " ", texto)
        texto = re.sub(r"\d+", " ", texto)
        texto = re.sub(r"\s+", " ", texto)
        return texto.strip()

    def tokenizar(self, texto):
        """Divide texto en tokens."""
        return nltk.word_tokenize(texto, language=self.idioma)

    def eliminar_stopwords(self, tokens):
        """Elimina palabras vacías."""
        return [t for t in tokens if t not in self.stopwords]

    def stemizar(self, tokens):
        """Aplica stemming."""
        return [self.stemmer.stem(t) for t in tokens]

    def procesar(self, texto):
        """Pipeline completo: limpiar -> tokenizar -> quitar stopwords -> stemizar."""
        texto_limpio = self.limpiar_texto(texto)
        tokens = self.tokenizar(texto_limpio)
        tokens_sin_stop = self.eliminar_stopwords(tokens)
        return self.stemizar(tokens_sin_stop)

    def analizar_corpus(self):
        """Genera estadísticas del corpus (una fila por texto)."""
        stats = []
        for i, texto in enumerate(self.textos):
            tokens_originales = self.tokenizar(texto)
            tokens_sin_stop = self.eliminar_stopwords(tokens_originales)
            tokens_procesados = self.stemizar(tokens_sin_stop)
            stats.append(
                {
                    "id": self.ids[i],
                    "categoria": self.etiquetas[i],
                    "palabras_originales": len(texto.split()),
                    "tokens": len(tokens_originales),
                    "tokens_sin_stopwords": len(tokens_sin_stop),
                    "palabras_unicas": len(set(tokens_procesados)),
                }
            )
        self.stats_df = pd.DataFrame(stats)
        return self.stats_df

    def visualizar_stats(self):
        """Visualiza estadísticas del corpus (distribución, longitud, diversidad)."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        self.stats_df["categoria"].value_counts().plot(
            kind="bar", ax=axes[0, 0], color="#4C72B0"
        )
        axes[0, 0].set_title("Distribución de textos por categoría")
        axes[0, 0].set_xlabel("Categoría")
        axes[0, 0].set_ylabel("Cantidad")

        self.stats_df.groupby("categoria")["palabras_originales"].mean().plot(
            kind="bar", ax=axes[0, 1], color="#DD8452"
        )
        axes[0, 1].set_title("Longitud promedio por categoría")
        axes[0, 1].set_xlabel("Categoría")
        axes[0, 1].set_ylabel("Promedio de palabras")

        sns.boxplot(data=self.stats_df, x="categoria", y="palabras_originales", ax=axes[1, 0])
        axes[1, 0].set_title("Distribución de longitudes")
        axes[1, 0].set_xlabel("Categoría")
        axes[1, 0].set_ylabel("Número de palabras")

        self.stats_df.groupby("categoria")["palabras_unicas"].mean().plot(
            kind="bar", ax=axes[1, 1], color="#55A868"
        )
        axes[1, 1].set_title("Diversidad de vocabulario por categoría")
        axes[1, 1].set_xlabel("Categoría")
        axes[1, 1].set_ylabel("Promedio de palabras únicas")

        plt.tight_layout()
        return fig
