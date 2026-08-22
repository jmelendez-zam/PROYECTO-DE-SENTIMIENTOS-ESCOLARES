# Análisis de Sentimientos en Respuestas Abiertas de Estudiantes

**Autor:** Jaime Meléndez Zambrano

Sistema completo de Procesamiento de Lenguaje Natural (NLP) que clasifica,
agrupa y genera texto a partir de respuestas abiertas de estudiantes sobre
su tiempo escolar, tiempo libre y carga académica, usando tres categorías
de sentimiento:

1. **Sentimientos placenteros** — bienestar, calma, satisfacción.
2. **Sentimientos desagradables** — cansancio, estrés, ansiedad, presión.
3. **Sentimientos mixtos** — respuestas con evidencia positiva y negativa a
   la vez (típicamente unidas por "pero"), por ejemplo *"me gusta la
   escuela, pero no me queda tiempo libre"*.

Este proyecto sigue la estructura de la asignación **"Laboratorio NLP:
Aprendizaje Supervisado, No Supervisado y Generativo"** (Fases 0 a 5), y
toma como guía de referencia (no como plantilla a copiar) el proyecto
[PROYECTO_NLP_ESTUDIANTE](https://github.com/18paz/PROYECTO_NLP_ESTUDIANTE),
que trabaja sobre el mismo tipo de datos con una clasificación binaria.
Aquí se usan **3 categorías** en vez de 2, con etiquetado hecho mediante un
lexicón propio (ver Fase 0).

## Origen de los datos: 12 encuestadores, no uno solo

La encuesta fue aplicada por **12 compañeros encuestadores**, cada uno a su
propio grupo de estudiantes. Sus respuestas crudas (`data/bases_de_datos/`,
830 archivos `.txt` en 5 formatos distintos) se consolidan en 965
respuestas individuales. Una versión anterior de este mismo proyecto,
sin darse cuenta, solo usaba los datos de **un** encuestador (200 archivos
de los 830 disponibles) porque su Excel de origen agrupaba las respuestas
en 2 celdas que resultaron ser, exactamente, las de ese encuestador — y esa
construcción tenía además un error real (dos respuestas fusionadas en una
por un salto de línea faltante). Esta versión reconstruye el corpus leyendo
directamente los 12 conjuntos de archivos crudos; el detalle completo está
en `notebooks/00_preparacion_corpus.ipynb`.

## Resultados principales

| Fase | Técnica | Resultado clave |
|:---|:---|:---|
| 0 | Etiquetado del corpus | 965 respuestas de 12 encuestadores; 843 etiquetadas con confianza (325 desagradable / 313 placentero / 205 mixto), 107 (11%) quedan para revisión manual |
| 1 | Preprocesamiento NLP | Pipeline de limpieza, tokenización, stopwords y stemming en español |
| 2 | Clasificación supervisada | Naive Bayes / Regresión Logística / SVM, 77.5-79.9% de precisión (3 clases) |
| 3 | Clustering (K-means) | k=3; silhouette score bajo en todo el rango probado (confirma poca estructura natural); coincidencia parcial (46-66%) con las categorías reales |
| 4 | Generación (Markov) | Orden 2 notablemente más coherente que orden 1; >2000 estados por categoría |
| 5 | Sistema integrado | Predicción + generación de estilo sobre 15 textos nunca vistos en entrenamiento; 73.3% de acierto en consenso |

## Estructura del repositorio

```
proyecto_sentimientos_escolares/
├── README.md
├── requirements.txt
├── data/
│   ├── bases_de_datos/                 # datos CRUDOS de los 12 encuestadores
│   │   ├── encuesta_abraham/           #   (5 formatos de archivo distintos,
│   │   ├── encuesta_eberto/            #    ver notebook 00 para el detalle)
│   │   ├── ... (12 carpetas en total)
│   │   └── encuesta_rosaura/
│   ├── datos_estudiantes.xlsx          # las 965 respuestas, vista tabular
│   ├── corpus_etiquetado.csv           # 843 textos etiquetados (entrenamiento)
│   ├── textos_revisar_manualmente.csv  # 107 textos sin evidencia léxica suficiente
│   ├── relatos_sin_etiqueta.csv        # 15 textos de prueba (Fase 5)
│   ├── fase4_textos_generados_markov.csv
│   ├── corpus_etiquetado/              # mismos 843 textos, en .txt por categoría
│   │   ├── placentero/      (313)
│   │   ├── desagradable/    (325)
│   │   └── mixto/           (205)
│   └── relatos_sin_etiqueta/           # los 15 textos de prueba en .txt
├── src/                                 # módulos de Python reutilizables
│   ├── etiquetador_sentimientos.py      # Fase 0 - lexicón de sentimientos
│   ├── construir_corpus.py              # Fase 0 - consolida las 12 fuentes crudas
│   ├── procesador_nlp.py                # Fase 1 - limpieza/tokenización/stats
│   ├── clasificador.py                  # Fase 2 - Naive Bayes/LR/SVM
│   ├── clusterizador.py                 # Fase 3 - K-means + silhouette
│   ├── generador_markov.py              # Fase 4 - cadenas de Markov
│   └── sistema_completo.py              # Fase 5 - integración
└── notebooks/
    ├── 00_preparacion_corpus.ipynb
    ├── 01_preprocesamiento.ipynb
    ├── 02_supervisado.ipynb
    ├── 03_no_supervisado.ipynb
    ├── 04_generativo.ipynb
    └── 05_integracion.ipynb
```

## Cómo ejecutar el proyecto

```bash
# 1. Clonar el repositorio
git clone <URL-de-tu-repositorio>
cd proyecto_sentimientos_escolares

# 2. Crear un entorno virtual (recomendado) e instalar dependencias
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Descargar los recursos de NLTK necesarios
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# 4. Abrir los notebooks en orden (00 -> 05)
jupyter notebook notebooks/
```

Los notebooks están pensados para ejecutarse **en orden**, ya que el
notebook `00_preparacion_corpus.ipynb` construye el corpus etiquetado
(`data/corpus_etiquetado/` y los `.csv`) a partir de `data/bases_de_datos/`,
que consumen los notebooks siguientes. Los seis notebooks ya vienen
ejecutados con sus salidas (tablas y gráficos) guardadas, por lo que
también pueden revisarse sin volver a correrlos.

## Metodología de etiquetado (Fase 0)

Los datos crudos (`data/bases_de_datos/`) solo traen la pregunta y la
respuesta de cada estudiante, **sin** etiqueta de sentimiento. Como
etiquetar manualmente 965 respuestas de forma consistente no era viable
para un proyecto individual, se construyó un etiquetador semi-automático
basado en un diccionario (lexicón) de palabras y frases en español
asociadas a cada sentimiento, con detección de conectores de contraste
("pero", "aunque") para identificar respuestas mixtas. El lexicón se
calibró en dos iteraciones contra el corpus completo (no solo contra una
muestra): la primera versión, calibrada únicamente contra los datos de un
encuestador, dejaba sin etiquetar el 27% de las 965 respuestas al aplicarla
a los 12 estilos de escritura distintos; tras ampliar el vocabulario y
corregir varios falsos positivos (verificando explícitamente que ningún
texto cambiara de polaridad de forma incorrecta), quedó en 11%. El método
completo, con su justificación, historial de calibración y limitaciones,
está documentado en el notebook `00_preparacion_corpus.ipynb` y en
`src/etiquetador_sentimientos.py` (cada etiqueta queda acompañada de la
evidencia léxica que la sustenta, por lo que el proceso es auditable y no
es una caja negra). El 11% de textos sin evidencia léxica suficiente se
excluye del corpus de entrenamiento pero se conserva, sin descartar, en
`data/textos_revisar_manualmente.csv`.

## Correcciones realizadas sobre el código guía de la tarea

Durante la implementación se detectó que el código de ejemplo de la guía
(`TfidfVectorizer(stop_words='spanish')`) no funciona: scikit-learn solo
acepta `'english'` como valor de cadena para `stop_words`, o una lista
explícita de palabras. Se corrigió pasando la lista de stopwords de NLTK
en español (ver `src/clasificador.py` y `src/clusterizador.py`).

Adicionalmente, al reconstruir el corpus a partir de las 12 fuentes crudas
se detectó que la versión anterior de este proyecto dependía de un Excel
intermedio frágil (dos celdas gigantes por pregunta, separadas por líneas
en blanco) que fusionó dos respuestas reales en una sola por un salto de
línea faltante en el archivo de origen. `src/construir_corpus.py` ya no
depende de ese paso intermedio: lee directamente los `.txt` crudos de cada
encuestador.

## Licencia de los datos

Los datos provienen de una encuesta educativa recolectada por 12
compañeros de curso (incluyendo al autor) y se usan exclusivamente con
fines académicos.
