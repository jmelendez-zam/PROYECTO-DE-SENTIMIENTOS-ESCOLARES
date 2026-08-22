"""
clasificador.py
----------------
Fase 2: Aprendizaje Supervisado.

Nota sobre una corrección respecto a la guía de la tarea:
La guía (tarea_proyecto.md) sugiere `TfidfVectorizer(stop_words='spanish')`,
pero scikit-learn SOLO acepta 'english' como valor de cadena para
`stop_words` (o una lista explícita de palabras); 'spanish' lanza
`ValueError: not a valid built-in stop list`. Aquí se corrige pasando
la lista de stopwords de NLTK en español directamente.
"""

import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

STOPWORDS_ES = list(stopwords.words("spanish"))


class ClasificadorTextos:
    def __init__(self, textos, etiquetas):
        self.textos = textos
        self.etiquetas = etiquetas
        self.vectorizer = None
        self.modelos = {}
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def vectorizar(self, max_features=2000):
        """Vectoriza textos usando TF-IDF (unigramas y bigramas).

        max_features=2000: con las ~843 respuestas del corpus completo el
        vocabulario unigrama+bigrama real es de ~5100 términos; 2000 cubre
        la gran mayoría de las combinaciones con TF-IDF más alto sin dejar
        crecer demasiado la dimensionalidad frente al número de ejemplos.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=STOPWORDS_ES,
            ngram_range=(1, 2),
        )
        X = self.vectorizer.fit_transform(self.textos)
        return X

    def dividir_datos(self, test_size=0.2, random_state=42):
        """Divide datos en entrenamiento y prueba (estratificado por categoría)."""
        X = self.vectorizar()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, self.etiquetas, test_size=test_size, random_state=random_state,
            stratify=self.etiquetas,
        )
        print(f"Entrenamiento: {self.X_train.shape[0]} textos")
        print(f"Prueba: {self.X_test.shape[0]} textos")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def entrenar_clasificadores(self):
        """Entrena Naive Bayes, Regresión Logística y SVM."""
        if self.X_train is None:
            self.dividir_datos()

        nb = MultinomialNB()
        nb.fit(self.X_train, self.y_train)
        self.modelos["Naive Bayes"] = nb

        lr = LogisticRegression(max_iter=2000, random_state=42)
        lr.fit(self.X_train, self.y_train)
        self.modelos["Logistic Regression"] = lr

        svm = SVC(kernel="linear", probability=True, random_state=42)
        svm.fit(self.X_train, self.y_train)
        self.modelos["SVM"] = svm

        print("Modelos entrenados:", list(self.modelos.keys()))
        return self.modelos

    def evaluar_modelos(self, verbose=True):
        """Evalúa todos los modelos entrenados sobre el conjunto de prueba."""
        resultados = {}
        for nombre, modelo in self.modelos.items():
            y_pred = modelo.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            reporte = classification_report(
                self.y_test, y_pred, output_dict=True, zero_division=0
            )
            resultados[nombre] = {
                "accuracy": accuracy,
                "report": reporte,
                "predictions": y_pred,
            }
            if verbose:
                print(f"\n=== {nombre} ===")
                print(f"Precisión: {accuracy:.3f}")
                print(classification_report(self.y_test, y_pred, zero_division=0))
        return resultados

    def matriz_confusion(self, nombre_modelo, etiquetas_orden=None):
        modelo = self.modelos[nombre_modelo]
        y_pred = modelo.predict(self.X_test)
        return confusion_matrix(self.y_test, y_pred, labels=etiquetas_orden)

    def top_features_por_clase(self, nombre_modelo="Logistic Regression", top_n=5):
        """Palabras con mayor peso por clase (solo para modelos lineales)."""
        modelo = self.modelos[nombre_modelo]
        features = np.array(self.vectorizer.get_feature_names_out())
        resultado = {}
        if hasattr(modelo, "coef_"):
            clases = modelo.classes_
            for i, clase in enumerate(clases):
                coefs = modelo.coef_[i] if len(clases) > 2 else modelo.coef_[0]
                top_idx = np.argsort(coefs)[-top_n:][::-1]
                resultado[clase] = list(features[top_idx])
        return resultado

    def predecir_nuevos(self, nuevos_textos):
        """Predice categoría de textos nuevos con todos los modelos entrenados."""
        if self.vectorizer is None:
            raise ValueError("Primero debes vectorizar los datos")

        X_nuevos = self.vectorizer.transform(nuevos_textos)
        predicciones = {}
        for nombre, modelo in self.modelos.items():
            pred = modelo.predict(X_nuevos)
            proba = modelo.predict_proba(X_nuevos) if hasattr(modelo, "predict_proba") else None
            predicciones[nombre] = {"categoria": pred, "probabilidades": proba}
        return predicciones
