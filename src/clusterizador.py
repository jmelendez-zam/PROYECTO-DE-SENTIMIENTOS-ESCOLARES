"""
clusterizador.py
-----------------
Fase 3: Aprendizaje No Supervisado (clustering).
"""

from collections import Counter

import numpy as np
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

STOPWORDS_ES = list(stopwords.words("spanish"))


class ClusterizadorTextos:
    def __init__(self, textos, vectorizer=None):
        self.textos = textos
        self.vectorizer = vectorizer
        self.X = None
        self.clusters = None
        self.kmeans = None

    def preparar_datos(self, max_features=2000):
        """Prepara los datos para clustering (TF-IDF, sin usar las etiquetas reales)."""
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words=STOPWORDS_ES)
        self.X = self.vectorizer.fit_transform(self.textos)
        return self.X

    def encontrar_k_optimo(self, max_k=10):
        """Método del codo: devuelve las inercias para k=2..max_k."""
        if self.X is None:
            self.preparar_datos()
        inertias = []
        for k in range(2, max_k + 1):
            modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
            modelo.fit(self.X)
            inertias.append(modelo.inertia_)
        return inertias

    def calcular_silhouette(self, max_k=10):
        """Calcula el coeficiente de silueta para k=2..max_k, como apoyo
        cuantitativo adicional al método del codo (inspirado en el
        proyecto guía, que usa silhouette_score para justificar su
        elección de k en la Fase 5). Un valor más alto (más cercano a 1)
        indica clusters mejor separados entre sí."""
        if self.X is None:
            self.preparar_datos()
        scores = {}
        for k in range(2, max_k + 1):
            modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
            etiquetas_k = modelo.fit_predict(self.X)
            scores[k] = silhouette_score(self.X, etiquetas_k)
        return scores

    def graficar_codo(self, inertias, max_k=10):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(9, 5))
        plt.plot(range(2, max_k + 1), inertias, marker="o", linewidth=2)
        plt.xlabel("Número de clusters (k)")
        plt.ylabel("Inercia")
        plt.title("Método del codo para k óptimo")
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        return fig

    def graficar_silhouette(self, scores):
        import matplotlib.pyplot as plt

        ks = list(scores.keys())
        valores = list(scores.values())
        fig = plt.figure(figsize=(9, 5))
        plt.plot(ks, valores, marker="o", linewidth=2, color="#DD8452")
        plt.xlabel("Número de clusters (k)")
        plt.ylabel("Coeficiente de silueta")
        plt.title("Coeficiente de silueta por número de clusters")
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        return fig

    def clusterizar(self, n_clusters=3):
        """Aplica K-means clustering."""
        if self.X is None:
            self.preparar_datos()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = self.kmeans.fit_predict(self.X)
        print(f"{n_clusters} clusters creados")
        for i in range(n_clusters):
            print(f"  Cluster {i}: {int(np.sum(self.clusters == i))} textos")
        return self.clusters

    def visualizar_clusters(self, etiquetas_reales=None):
        """Visualiza los clusters en 2D mediante PCA."""
        import matplotlib.pyplot as plt

        reducer = PCA(n_components=2, random_state=42)
        X_reduced = reducer.fit_transform(self.X.toarray())

        colores = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
        fig, axes = plt.subplots(1, 2 if etiquetas_reales is not None else 1, figsize=(14, 6))
        if etiquetas_reales is None:
            axes = [axes]

        ax = axes[0]
        for i in range(max(self.clusters) + 1):
            mask = self.clusters == i
            ax.scatter(
                X_reduced[mask, 0], X_reduced[mask, 1],
                c=colores[i % len(colores)], label=f"Cluster {i}", alpha=0.75, s=70,
            )
        ax.set_title("Clusters encontrados (K-means)")
        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if etiquetas_reales is not None:
            ax2 = axes[1]
            categorias = sorted(set(etiquetas_reales))
            for i, cat in enumerate(categorias):
                mask = np.array(etiquetas_reales) == cat
                ax2.scatter(
                    X_reduced[mask, 0], X_reduced[mask, 1],
                    c=colores[i % len(colores)], label=cat, alpha=0.75, s=70,
                )
            ax2.set_title("Categorías reales de sentimiento")
            ax2.set_xlabel("Componente 1")
            ax2.set_ylabel("Componente 2")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def analizar_clusters(self, etiquetas_reales=None):
        """Analiza palabras características y categoría real predominante de cada cluster."""
        feature_names = self.vectorizer.get_feature_names_out()
        analisis = {}
        for i in range(max(self.clusters) + 1):
            centroide = self.kmeans.cluster_centers_[i]
            top_indices = np.argsort(centroide)[-10:][::-1]
            top_palabras = [feature_names[idx] for idx in top_indices]

            textos_cluster = [j for j, c in enumerate(self.clusters) if c == i]

            categoria_principal = None
            coincidencia = None
            if etiquetas_reales is not None:
                categorias_cluster = [etiquetas_reales[j] for j in textos_cluster]
                if categorias_cluster:
                    conteo = Counter(categorias_cluster)
                    categoria_principal, cantidad = conteo.most_common(1)[0]
                    coincidencia = cantidad / len(categorias_cluster)

            analisis[i] = {
                "num_textos": len(textos_cluster),
                "top_palabras": top_palabras,
                "textos_indices": textos_cluster,
                "categoria_principal": categoria_principal,
                "coincidencia": coincidencia,
            }
        return analisis
