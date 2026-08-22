# Informe Técnico — Análisis de Sentimientos en Respuestas Abiertas de Estudiantes

**Autor:** Jaime Meléndez Zambrano

---

## 1. Resumen del proyecto

Este proyecto construye un sistema completo de NLP sobre un corpus de 965
respuestas abiertas de estudiantes a una encuesta sobre tiempo escolar,
tiempo libre y carga académica, recolectadas por **12 compañeros
encuestadores**. El sistema clasifica cada respuesta en una de tres
categorías de sentimiento (**placentero**, **desagradable**, **mixto**),
agrupa las respuestas sin usar las etiquetas (clustering), y genera texto
nuevo de estilo similar mediante cadenas de Markov. El desarrollo completo,
con código y resultados, está documentado fase por fase en los notebooks
`notebooks/00` a `notebooks/05`; este documento resume la metodología y las
conclusiones principales.

Esta versión del proyecto reconstruye el corpus desde cero a partir de los
datos crudos de los 12 encuestadores (`data/bases_de_datos/`). Una versión
anterior, sin darse cuenta, solo usaba los datos de un encuestador (200 de
830 archivos disponibles), con un error adicional de construcción que
fusionaba dos respuestas en una. La sección 2 detalla cómo se identificó y
corrigió esto.

## 2. Datos y etiquetado (Fase 0)

### 2.1 Consolidación de 12 fuentes con 5 formatos distintos

`data/bases_de_datos/` contiene 830 archivos `.txt` crudos, uno por
respuesta o por estudiante según el encuestador. Se identificaron 5
convenciones de formato distintas entre los 12 encuestadores (un archivo
por respuesta con el número de pregunta en el nombre; dos respuestas por
archivo como oraciones seguidas; dos respuestas separadas por línea en
blanco; una respuesta general sin separar por pregunta; y un formato
CSV de una línea con 4 campos). Cada patrón de parseo se validó contra el
**100% de los archivos** de su carpeta correspondiente (no una muestra)
antes de aceptarlo, lo que dio como resultado 965 respuestas individuales
sin pérdida ni mezcla incorrecta de datos.

### 2.2 Etiquetado semi-automático

Al no existir una etiqueta de sentimiento preexistente, se construyó un
etiquetador basado en un diccionario de palabras y frases en español
(`src/etiquetador_sentimientos.py`), que detecta:

- Vocabulario de bienestar (*bien, feliz, tranquilo, disfrutar...*) y de
  malestar (*cansado, estresado, ansiedad, agobiado...*).
- Frases que indican carencia o insatisfacción (*no me alcanza el tiempo,
  no tengo tiempo, me gustaría, quisiera...*).
- Conectores de contraste (*pero, aunque, sin embargo*) que, combinados
  con evidencia positiva y negativa a la vez, marcan una respuesta como
  **mixta**.

**Calibración contra el corpus completo.** La primera versión del lexicón
se había calibrado únicamente contra los datos de un encuestador (199
textos). Al aplicarla, sin cambios, a los 965 textos de los 12
encuestadores —con estilos de escritura y modismos regionales distintos
("chévere", "bacano", "flojera")— el 27% de los textos quedaba sin
evidencia léxica suficiente para etiquetarse. Se amplió el lexicón en dos
iteraciones, cada una seguida de una comparación explícita de la etiqueta
de **todo** el corpus antes y después del cambio, para detectar falsos
positivos introducidos por el vocabulario nuevo (por ejemplo, la palabra
"bueno" usada como muletilla discursiva al inicio de una frase, o
negaciones no capturadas como "no es excesivo"). Tras las dos iteraciones,
el porcentaje sin resolver bajó a 11%, sin introducir ninguna reversión de
polaridad no verificada.

Resultado final: **330 respuestas desagradables, 318 placenteras, 210
mixtas** (965 totales) y **107 (11%) sin evidencia léxica suficiente**, que
se excluyen del entrenamiento pero se conservan en
`data/textos_revisar_manualmente.csv` para eventual revisión humana. Se
separaron 15 respuestas (5 por categoría, con su etiqueta real oculta) como
conjunto de prueba para la Fase 5, dejando **843** para el entrenamiento de
las fases 1-4.

### 2.3 Calidad del corpus

Se verificó codificación UTF-8 (830/830 archivos crudos sin error), ausencia
de archivos vacíos, y duplicados exactos: 41 grupos (149 respuestas),
concentrados sobre todo en el encuestador `eberto` (cuyos archivos, cada 5
estudiantes, repiten el mismo texto — posible reutilización de una
respuesta de ejemplo). Siguiendo el mismo criterio metodológico del
proyecto guía ("no se modificará el texto original"), estos duplicados no
se eliminan: se documentan por transparencia.

## 3. Preprocesamiento (Fase 1)

Pipeline estándar de NLP en español con NLTK: normalización (minúsculas,
sin puntuación ni números, conservando tildes/ñ), tokenización,
eliminación de *stopwords* y *stemming* (`SnowballStemmer`). Se
representa el corpus con TF-IDF (unigramas y bigramas; `max_features=2000`,
ajustado desde 1000 porque el vocabulario real del corpus ampliado ronda
los 5100 términos únicos). Las respuestas promedian 18 palabras; la
categoría "mixto" tiene, en promedio, las respuestas más largas (23
palabras) — consistente con que necesita espacio para expresar evidencia
positiva y negativa a la vez.

## 4. Clasificación supervisada (Fase 2)

Se entrenaron tres modelos (Naive Bayes, Regresión Logística, SVM lineal)
sobre TF-IDF, con una división 80/20 estratificada (674 textos de
entrenamiento, 169 de prueba). La precisión se ubicó entre **0.775 y
0.799**, notablemente superior al 0.641-0.667 de la versión con un solo
encuestador. La categoría "desagradable" sigue siendo la más fácil de
reconocer; "placentero" dejó de ser la más difícil (pasó de 16 a 313
ejemplos de entrenamiento) y ahora es "mixto" la que concentra más
confusión, por definición: comparte vocabulario con ambas categorías
puras.

**Corrección aplicada:** el código guía sugiere
`TfidfVectorizer(stop_words='spanish')`, que produce un error en
scikit-learn (solo acepta `'english'` como cadena). Se corrigió pasando
la lista de stopwords de NLTK en español.

## 5. Clustering no supervisado (Fase 3)

K-means con k=3 (elegido para comparar directamente contra las 3
categorías reales). Como apoyo cuantitativo adicional al método del codo
—inspirado en el uso de `silhouette_score` del proyecto guía— se calculó
el coeficiente de silueta para k=2 a 8: los valores se mantuvieron bajos en
todo el rango (0.02-0.03), confirmando numéricamente que el corpus no tiene
una estructura de agrupamiento fuerte en el espacio TF-IDF. La coincidencia
entre clusters y categorías reales es parcial (46%-66%): los clusters
reflejan más el **tema** de la respuesta (tiempo libre vs. carga académica)
que su **polaridad emocional**.

## 6. Modelos generativos (Fase 4)

Cadenas de Markov de orden 1 y 2, entrenadas por separado para cada
categoría (ahora con 205 a 325 textos de entrenamiento por categoría, frente
a 16-109 antes). El orden 2 sigue generando texto notablemente más
coherente que el orden 1, y con más de 2000 estados distintos por categoría
(antes 102-378) las generaciones son menos repetitivas entre sí.

## 7. Integración (Fase 5)

El sistema completo (`src/sistema_completo.py`) conecta las tres técnicas:
dado un texto nuevo, predice su categoría de sentimiento y usa el
generador Markov de esa categoría para producir una continuación de
estilo similar. Puesto a prueba contra 15 textos de prueba (5 por
categoría, nunca vistos en entrenamiento), el consenso de los
clasificadores acertó en **11 de 15 (73.3%)**, una mejora frente al 4/6
(66.7%) de la versión con un solo encuestador y una base de comparación
más representativa (las tres categorías están igualmente presentes en la
prueba, no solo mayoritariamente "desagradable").

## 8. Conclusiones generales

- Reconstruir el corpus a partir de las **12 fuentes disponibles**, en vez
  de solo una, tuvo un efecto medible y positivo en cada fase posterior:
  más datos, mejor balance de clases, y mejor desempeño de los
  clasificadores — sin necesidad de cambiar el enfoque técnico.
- Gran parte de las limitaciones reportadas en la versión anterior (bajo
  desempeño en "placentero", fuerte desbalance hacia "desagradable") eran
  consecuencia de usar los datos de un solo encuestador, no una limitación
  inherente del pipeline.
- Un corpus recolectado por múltiples personas, con formatos de archivo
  distintos entre sí, requiere una etapa de consolidación cuidadosa
  (Fase 0) que es tan importante como el modelado posterior: un error de
  origen ahí (como el Excel que fusionaba dos respuestas) se propaga
  silenciosamente a todas las fases siguientes si no se detecta.
- La categoría "mixto" resultó indispensable: sigue siendo una fracción
  sustancial del corpus (24%) y las respuestas que la componen son, en
  promedio, las más largas — consistente con la idea de que expresar dos
  sentimientos a la vez requiere más palabras que expresar uno solo.
- Las tres técnicas (clasificación, clustering, generación) capturan
  aspectos distintos y complementarios del corpus: polaridad emocional,
  estructura temática y estilo lingüístico, respectivamente.

Para el detalle completo de código, tablas y gráficos de cada fase, ver
los notebooks en `notebooks/00_preparacion_corpus.ipynb` a
`notebooks/05_integracion.ipynb`.
