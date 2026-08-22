"""
construir_corpus.py
--------------------
Construye el corpus etiquetado del proyecto a partir de las respuestas
crudas de encuesta guardadas en `data/bases_de_datos/`. Este script se
ejecuta desde el notebook de la Fase 0 y deja materializados en disco:

  data/datos_estudiantes.xlsx                -> las ~965 respuestas individuales
                                                 (una fila por respuesta), en
                                                 formato tabular de conveniencia
  data/corpus_etiquetado.csv                 -> respuestas ETIQUETADAS con
                                                 confianza (entrenamiento)
  data/corpus_etiquetado/<categoria>/*.txt   -> un archivo .txt por respuesta
                                                 (misma estructura de carpetas
                                                 que el corpus de ejemplo de la guía)
  data/relatos_sin_etiqueta/*.txt            -> conjunto de prueba (con etiqueta
                                                 real oculta) para la Fase 5
  data/relatos_sin_etiqueta.csv              -> el mismo conjunto de prueba,
                                                 en formato tabular, con la
                                                 etiqueta real para autoevaluación
  data/textos_revisar_manualmente.csv        -> respuestas que el lexicón no
                                                 pudo etiquetar con confianza
                                                 (evidencia positiva y negativa
                                                 nula o empatada); se excluyen
                                                 del corpus de entrenamiento por
                                                 transparencia, no se descartan.

Origen de los datos
====================
`data/bases_de_datos/` contiene las respuestas **crudas** recolectadas por
12 compañeros encuestadores (cada quien encuestó a un grupo distinto de
estudiantes), cada uno con su propia convención de nombres y formato de
archivo. Se identificaron 5 formatos distintos y se validó cada uno
contra el 100% de sus archivos (no solo una muestra) antes de asumirlo:

  A. Un archivo = una respuesta; el número de pregunta va en el nombre
     del archivo (p.ej. `edgar_pregunta1_respuesta003.txt`).
     Encuestadores: abraham, edgar, elvis, jair, juan, karen, menco.
  B. Un archivo = 2 respuestas (pregunta 1 y 2) como 2 oraciones
     seguidas, sin salto de línea entre ellas.
     Encuestadores: eberto, jaime.
  C. Un archivo = 2 respuestas separadas por una línea en blanco.
     Encuestador: marlon.
  D. Un archivo = 1 respuesta general que mezcla ambos temas sin
     separación clara (no se fuerza una división artificial).
     Encuestador: jorge.
  E. Un archivo = 1 línea con 4 campos separados por `;`
     (índice; código de estudiante; respuesta 1; respuesta 2).
     Encuestador: rosaura.

No se corrigen errores de ortografía ni de tipeo de los estudiantes
(p. ej. "apr ndo" en vez de "aprendo"): son parte del texto original y
alterarlo sería introducir un sesgo del investigador. Sí se normalizan
espacios en blanco internos (saltos de línea sueltos dentro de una
misma respuesta) para que la longitud en palabras se calcule bien.
"""

import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from etiquetador_sentimientos import etiquetar  # noqa: E402

RUTA_BASE = os.path.join(os.path.dirname(__file__), "..", "data")
RUTA_BASES_DE_DATOS = os.path.join(RUTA_BASE, "bases_de_datos")
RUTA_XLSX = os.path.join(RUTA_BASE, "datos_estudiantes.xlsx")
RUTA_CORPUS = os.path.join(RUTA_BASE, "corpus_etiquetado")
RUTA_SIN_ETIQUETA = os.path.join(RUTA_BASE, "relatos_sin_etiqueta")
RUTA_REVISAR = os.path.join(RUTA_BASE, "textos_revisar_manualmente.csv")

PREGUNTA1_TXT = (
    "¿Cómo te sientes respecto al tiempo que pasas en la escuela y el "
    "tiempo libre que te queda en casa?"
)
PREGUNTA2_TXT = (
    "¿Qué opinión tienes sobre el volumen de tareas, proyectos y "
    "evaluaciones académicas que recibes?"
)
PREGUNTA_GENERAL_TXT = (
    "Respuesta general (mezcla ambos temas, sin separación clara en el "
    "archivo original)"
)

# Encuestadores con formato A: un archivo = una respuesta, con el número
# de pregunta codificado en el nombre del archivo.
FORMATO_A = {
    "encuesta_abraham": (r"^pregunta([12])_estudiante(\d+)\.txt$", "abraham"),
    "encuesta_edgar": (r"^edgar_pregunta([12])_respuesta(\d+)\.txt$", "edgar"),
    "encuesta_elvis": (r"^elvis_pregunta([12])_respuesta(\d+)\.txt$", "elvis"),
    "encuesta_jair": (r"^jair_pregunta([12])_respuesta(\d+)\.txt$", "jair"),
    "encuesta_juan": (r"^jpmh_pregunta([12])_estudiante(\d+)\.txt$", "juan"),
    "encuesta_karen": (r"^karen_pregunta([12])_respuesta(\d+)\.txt$", "karen"),
    "encuesta_menco": (r"^menco_pregunta([12])_respuesta(\d+)\.txt$", "menco"),
}

# Encuestadores con formato B: 2 oraciones seguidas en un mismo archivo.
FORMATO_B = {
    "encuesta_eberto": "eberto",
    "encuesta_jaime": "jaime",
}

# Encuestadores con formato C: 2 párrafos separados por línea en blanco.
FORMATO_C = {
    "encuesta_marlon": "marlon",
}

# Encuestadores con formato D: respuesta única, sin separar por pregunta.
FORMATO_D = {
    "encuesta_jorge": "jorge",
}

# Encuestadores con formato E: 1 línea CSV "indice;codigo;resp1;resp2".
FORMATO_E = {
    "encuesta_rosaura": "rosaura",
}

# Cuántos textos de cada categoría se retiran del corpus de entrenamiento
# para usarlos como "relatos sin etiqueta" (prueba final, Fase 5).
HOLDOUT_POR_CATEGORIA = {"desagradable": 5, "placentero": 5, "mixto": 5}
SEED = 42


def _normalizar_espacios(texto: str) -> str:
    return re.sub(r"\s+", " ", texto).strip()


def _leer_formato_a(carpeta_encuestador: str) -> list:
    patron, encuestador = FORMATO_A[os.path.basename(carpeta_encuestador)]
    registros = []
    for archivo in sorted(os.listdir(carpeta_encuestador)):
        m = re.match(patron, archivo)
        if not m:
            continue
        num_pregunta = m.group(1)
        with open(os.path.join(carpeta_encuestador, archivo), encoding="utf-8") as f:
            texto = f.read().strip()
        registros.append(_registro(encuestador, archivo, num_pregunta, texto))
    return registros


def _leer_formato_b(carpeta_encuestador: str) -> list:
    encuestador = FORMATO_B[os.path.basename(carpeta_encuestador)]
    registros = []
    for archivo in sorted(os.listdir(carpeta_encuestador)):
        if not archivo.endswith(".txt"):
            continue
        with open(os.path.join(carpeta_encuestador, archivo), encoding="utf-8") as f:
            contenido = f.read().strip()
        partes = [p.strip() for p in re.split(r"(?<=[.])\s+", contenido) if p.strip()]
        for i, texto in enumerate(partes[:2], start=1):
            registros.append(_registro(encuestador, archivo, str(i), texto))
    return registros


def _leer_formato_c(carpeta_encuestador: str) -> list:
    encuestador = FORMATO_C[os.path.basename(carpeta_encuestador)]
    registros = []
    for archivo in sorted(os.listdir(carpeta_encuestador)):
        if not archivo.endswith(".txt"):
            continue
        with open(os.path.join(carpeta_encuestador, archivo), encoding="utf-8") as f:
            contenido = f.read()
        partes = [p.strip() for p in contenido.split("\n\n") if p.strip()]
        for i, texto in enumerate(partes[:2], start=1):
            registros.append(_registro(encuestador, archivo, str(i), texto))
    return registros


def _leer_formato_d(carpeta_encuestador: str) -> list:
    encuestador = FORMATO_D[os.path.basename(carpeta_encuestador)]
    registros = []
    for archivo in sorted(os.listdir(carpeta_encuestador)):
        if not archivo.endswith(".txt"):
            continue
        with open(os.path.join(carpeta_encuestador, archivo), encoding="utf-8") as f:
            texto = f.read().strip()
        registros.append(_registro(encuestador, archivo, "general", texto))
    return registros


def _leer_formato_e(carpeta_encuestador: str) -> list:
    encuestador = FORMATO_E[os.path.basename(carpeta_encuestador)]
    registros = []
    for archivo in sorted(os.listdir(carpeta_encuestador)):
        if not archivo.endswith(".txt"):
            continue
        with open(os.path.join(carpeta_encuestador, archivo), encoding="utf-8") as f:
            contenido = f.read().strip()
        campos = contenido.split(";")
        if len(campos) != 4:
            continue
        resp1, resp2 = campos[2].strip(), campos[3].strip()
        for i, texto in enumerate([resp1, resp2], start=1):
            registros.append(_registro(encuestador, archivo, str(i), texto))
    return registros


def _registro(encuestador, archivo, num_pregunta, texto) -> dict:
    texto = _normalizar_espacios(texto)
    if num_pregunta == "1":
        pregunta_origen = PREGUNTA1_TXT
    elif num_pregunta == "2":
        pregunta_origen = PREGUNTA2_TXT
    else:
        pregunta_origen = PREGUNTA_GENERAL_TXT
    return {
        "encuestador": encuestador,
        "archivo_origen": archivo,
        "pregunta_num": num_pregunta,
        "pregunta_origen": pregunta_origen,
        "texto": texto,
    }


def cargar_respuestas_individuales(ruta_bases_de_datos: str = RUTA_BASES_DE_DATOS) -> pd.DataFrame:
    """Lee las 12 carpetas de encuestadores (5 formatos distintos) y
    devuelve un DataFrame con una fila por respuesta individual."""
    lectores = {}
    for carpeta in FORMATO_A:
        lectores[carpeta] = _leer_formato_a
    for carpeta in FORMATO_B:
        lectores[carpeta] = _leer_formato_b
    for carpeta in FORMATO_C:
        lectores[carpeta] = _leer_formato_c
    for carpeta in FORMATO_D:
        lectores[carpeta] = _leer_formato_d
    for carpeta in FORMATO_E:
        lectores[carpeta] = _leer_formato_e

    registros = []
    for carpeta_nombre, funcion_lectura in sorted(lectores.items()):
        ruta_carpeta = os.path.join(ruta_bases_de_datos, carpeta_nombre)
        registros.extend(funcion_lectura(ruta_carpeta))

    df = pd.DataFrame(registros)
    df = df[df["texto"].str.len() > 0].reset_index(drop=True)
    df["id"] = [f"texto_{i:04d}" for i in range(len(df))]
    df["num_palabras"] = df["texto"].apply(lambda t: len(t.split()))
    columnas = ["id", "encuestador", "archivo_origen", "pregunta_num",
                "pregunta_origen", "texto", "num_palabras"]
    return df[columnas]


def etiquetar_corpus(df: pd.DataFrame) -> pd.DataFrame:
    resultados = df["texto"].apply(etiquetar)
    df = df.copy()
    df["etiqueta"] = [r["etiqueta"] for r in resultados]
    df["pos_score"] = [r["pos_score"] for r in resultados]
    df["neg_score"] = [r["neg_score"] for r in resultados]
    df["evidencia_pos"] = [", ".join(r["evidencia_pos"]) for r in resultados]
    df["evidencia_neg"] = [", ".join(r["evidencia_neg"]) for r in resultados]
    return df


def separar_holdout(df: pd.DataFrame):
    """Separa el holdout de Fase 5 y descarta 'revisar_manualmente' del
    corpus de entrenamiento (se guardan aparte, no se botan)."""
    df_confiable = df[df["etiqueta"] != "revisar_manualmente"].reset_index(drop=True)
    df_revisar = df[df["etiqueta"] == "revisar_manualmente"].reset_index(drop=True)

    holdout_partes = []
    for categoria, n in HOLDOUT_POR_CATEGORIA.items():
        sub = df_confiable[df_confiable["etiqueta"] == categoria].sample(
            n=n, random_state=SEED
        )
        holdout_partes.append(sub)
    holdout = pd.concat(holdout_partes)
    entrenamiento = df_confiable.drop(holdout.index).reset_index(drop=True)
    holdout = holdout.reset_index(drop=True)
    return entrenamiento, holdout, df_revisar


def guardar_txt_por_categoria(df: pd.DataFrame, ruta_corpus: str):
    for categoria in df["etiqueta"].unique():
        os.makedirs(os.path.join(ruta_corpus, categoria), exist_ok=True)
    for _, fila in df.iterrows():
        ruta = os.path.join(ruta_corpus, fila["etiqueta"], f"{fila['id']}.txt")
        with open(ruta, "w", encoding="utf-8") as f:
            f.write(fila["texto"])


def guardar_txt_sin_etiqueta(df: pd.DataFrame, ruta: str):
    os.makedirs(ruta, exist_ok=True)
    for _, fila in df.iterrows():
        with open(os.path.join(ruta, f"{fila['id']}.txt"), "w", encoding="utf-8") as f:
            f.write(fila["texto"])


def guardar_excel_original(df_raw: pd.DataFrame, ruta_xlsx: str):
    """Guarda una vista tabular de conveniencia de TODAS las respuestas
    crudas (sin etiquetar), una fila por respuesta, con su encuestador y
    pregunta de origen. Reemplaza el formato anterior (2 celdas gigantes
    separadas por líneas en blanco), que era más frágil: un solo salto de
    línea faltante en el archivo fuente fusionaba dos respuestas en una."""
    os.makedirs(os.path.dirname(ruta_xlsx), exist_ok=True)
    df_raw.to_excel(ruta_xlsx, index=False)


def construir(guardar: bool = True) -> dict:
    df_raw = cargar_respuestas_individuales(RUTA_BASES_DE_DATOS)
    df = etiquetar_corpus(df_raw)
    entrenamiento, holdout, revisar = separar_holdout(df)

    if guardar:
        os.makedirs(RUTA_BASE, exist_ok=True)
        guardar_excel_original(df_raw, RUTA_XLSX)
        entrenamiento.to_csv(os.path.join(RUTA_BASE, "corpus_etiquetado.csv"), index=False)
        holdout.to_csv(os.path.join(RUTA_BASE, "relatos_sin_etiqueta.csv"), index=False)
        revisar.to_csv(RUTA_REVISAR, index=False)
        guardar_txt_por_categoria(entrenamiento, RUTA_CORPUS)
        guardar_txt_sin_etiqueta(holdout, RUTA_SIN_ETIQUETA)

    return {
        "df_raw": df_raw,
        "df_completo": df,
        "entrenamiento": entrenamiento,
        "holdout": holdout,
        "revisar": revisar,
    }


if __name__ == "__main__":
    resultado = construir(guardar=True)
    print("Total respuestas individuales (crudas):", len(resultado["df_raw"]))
    print("\nDistribución (corpus completo, incluye revisar_manualmente):")
    print(resultado["df_completo"]["etiqueta"].value_counts())
    print("\nEntrenamiento:", len(resultado["entrenamiento"]))
    print(resultado["entrenamiento"]["etiqueta"].value_counts())
    print("\nHoldout (relatos sin etiqueta):", len(resultado["holdout"]))
    print(resultado["holdout"]["etiqueta"].value_counts())
    print("\nPara revisión manual (excluidos del entrenamiento):", len(resultado["revisar"]))
