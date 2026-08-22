"""
etiquetador_sentimientos.py
----------------------------
Etiquetador semi-automático de sentimientos para la Fase 0 del proyecto.

Contexto
========
El corpus original (respuestas abiertas de estudiantes a una encuesta
sobre tiempo escolar / tiempo libre / carga académica) no trae una
etiqueta de sentimiento: solo trae la pregunta y el texto de respuesta.

Como el equipo no puede leer y clasificar manualmente casi mil
respuestas de forma consistente, se construyó un etiquetador basado en
un diccionario (lexicón) de palabras y frases asociadas a:

  - sentimientos PLACENTEROS  (bienestar, calma, disfrute)
  - sentimientos DESAGRADABLES (cansancio, estrés, ansiedad, presión)
  - conectores de CONTRASTE ("pero", "aunque", "sin embargo") que,
    combinados con evidencia positiva Y negativa en el mismo texto,
    indican un sentimiento MIXTO.

Este método es un *punto de partida reproducible* (no una verdad
absoluta): en la Fase 0 se documentan sus reglas y limitaciones, y se
recomienda una revisión manual de los casos límite (empates o
puntajes muy bajos) antes de usar el corpus en un contexto distinto
al de este proyecto académico.

Historial de calibración
=========================
La primera versión del lexicón se calibró únicamente contra los 199
textos de un solo encuestador. Al ampliar el corpus a los 965 textos
de los 12 encuestadores de `data/bases_de_datos/` (12 estilos de
escritura distintos, con modismos regionales como "chévere", "bacano"
o "flojera"), esa primera versión dejaba sin resolver ("revisar
manualmente") cerca de 27% de los textos. Se amplió el vocabulario en
dos iteraciones, revisando en cada una:

  1. una muestra de los textos que quedaban en "revisar_manualmente"
     (para añadir vocabulario que faltaba: "chévere", "bacano",
     "flojera", "aburrido", "estresante", "difícil", "conforme",
     "adecuado", "fácil", entre otros), y
  2. un diff palabra por palabra entre la etiqueta antes/después de
     cada cambio, para detectar y corregir **falsos positivos**
     introducidos por el vocabulario nuevo (p. ej. "bueno" como
     muletilla al inicio de una frase -"Bueno, la verdad..."-, o
     negaciones no capturadas como "no hay tanta tarea acumulada" /
     "no considero que sea excesivo").

Con esas dos iteraciones, el porcentaje de textos sin resolver bajó de
~27% a ~11%, sin introducir ninguna reversión de polaridad (ningún
texto pasó de "placentero" a "desagradable" o viceversa sin que fuera
una corrección deliberada de un error real). El ~11% restante
(mayoritariamente respuestas muy cortas y genuinamente neutras como
"normal" o "más o menos", sin ningún otro contenido) se deja
honestamente fuera del corpus de entrenamiento en vez de forzar una
etiqueta: ver `data/textos_revisar_manualmente.csv` y la sección
"Calidad y validación del corpus" en el notebook de la Fase 0.

Regla de decisión
==================
  1. Se buscan primero las FRASES_NEGATIVAS, FRASES_POSITIVAS y
     FRASES_NEUTRALIZADORAS (más específicas y confiables que una
     palabra suelta) y se "apagan" esos tramos de texto para que sus
     palabras no se vuelvan a contar sueltas. Las neutralizadoras
     (p. ej. "no es excesivo") no suman puntaje a ningún lado: solo
     evitan que la palabra negada dentro de ellas ("excesivo") se
     cuente como evidencia negativa suelta.
  2. Sobre el texto restante se cuentan palabras de POS_WORDS y
     NEG_WORDS.
  3. Si hay evidencia positiva Y negativa a la vez -> "mixto".
  4. Si predomina una sola polaridad -> esa categoría.
  5. Si no hay ninguna evidencia -> se marca como "revisar_manualmente"
     para inspección manual (en este corpus, tras el ajuste del
     lexicón, queda un ~11% de casos así; ver nota de calibración).
"""

import re

# Frases que indican negación/insatisfacción -> suman a neg_score y se "apagan"
FRASES_NEGATIVAS = [
    r"no\s+(me\s+)?(tengo|queda|alcanza|permite|deja|sobra)\w*",
    r"no\s+(son|es)\s+(justas?|suficiente\w*|necesarias?)",
    r"no\s+tienen?\s+sentido",
    r"no\s+aportan?",
    r"no\s+puedo\w*",
    r"no\s+s[ée]\s+por",
    r"no\s+duermo\w*",
    r"no\s+existe\w*",
    r"no\s+rinde\w*",
    r"nunca\s+alcanz\w*",
    r"casi\s+no\s+\w+",
    r"casi\s+nunca\s*\w*",
    r"me\s+gustar[ií]a",
    r"\bquisiera\b",
    r"desear[ií]a",
    r"deber[ií]a(n|mos)?\s+\w+",
    r"(solo\s+)?quiero\s+(dormir|desconectarme)",
    r"me\s+rindo\w*",
    r"falta\s+de\s+respeto",
    r"no\s+me\s+siento\s+(muy\s+)?a\s+gusto",
]

# Frases que indican satisfacción/logro -> suman a pos_score y se "apagan"
FRASES_POSITIVAS = [
    r"vale\s+la\s+pena",
    r"me\s+desestres\w*",
]

# Frases que NEUTRALIZAN una palabra de polaridad que aparecería justo después
# (negación de un término negativo). Se "apagan" pero NO suman a ningún score,
# para que la palabra negada (p.ej. "excesivo" en "no es excesivo") no se
# cuente como evidencia negativa suelta.
FRASES_NEUTRALIZADORAS = [
    r"no\s+hay\s+tant\w+\s+\w+\s+acumulad\w*",
    r"no\s+se\s+acumula\w*",
    r"no\s+(es|son|parece)\s+excesiv\w*",
    r"no\s+(considero|creo|pienso)\s+que\s+sea\s+excesiv\w*",
    r"no\s+me\s+afecta\s+demasiado",
]

POS_WORDS = {
    "bien", "feliz", "felices", "tranquilo", "tranquila", "equilibrado", "equilibrada",
    "agradable", "cómodo", "cómoda", "comodo", "comoda", "contento", "contenta",
    "satisfecho", "satisfecha", "disfruto", "disfrutar", "organizo", "aprendo", "aprender",
    "encanta", "gusta", "importante", "importantes", "necesario", "necesaria",
    "necesarios", "necesarias", "fundamental", "cumplir", "manejar",
    "interesante", "interesantes",
    # -- ampliación v2/v3 (calibrada contra el corpus completo de 12 encuestadores) --
    "buen", "bueno", "buena", "buenos", "buenas",
    "chevere", "chévere", "cheveres", "chéveres", "bacano", "bacana", "bacanos", "bacanas",
    "conforme", "adecuado", "adecuada", "adecuados", "adecuadas", "apropiado", "apropiada",
    "relajado", "relajada", "relajados", "relajadas", "relajo", "calmado", "calmada",
    "divertido", "divertida", "divertidos", "divertidas", "diversion", "diversión", "firme",
    "aprovecho", "fortalece", "fortalecen", "refuerza", "refuerzan",
    "beneficia", "beneficial", "beneficioso", "beneficiosa", "benefician",
    "pasable", "aceptable", "activo", "activa", "equilibrio",
    "facil", "fácil", "faciles", "fáciles", "primordial", "jugar", "juego", "gusto",
}

NEG_WORDS = {
    "cansado", "cansada", "cansa", "cansan", "agobiado", "agobiada", "agotado", "agotada",
    "estresado", "estresada", "estrés", "estres", "ansiedad", "ansioso", "ansiosa",
    "frustra", "frustración", "frustracion", "frustrado", "frustrada", "atrapado", "atrapada",
    "abrumado", "abrumada", "nervioso", "nerviosa", "presión", "presion", "sobrecarga",
    "excesivo", "excesiva", "excesivos", "excesivas", "demasiado", "demasiada",
    "demasiados", "demasiadas", "escaso", "escasa", "injusta", "injustas",
    "rígido", "rigido", "rígida", "rigida", "pesado", "pesada", "quita", "roba",
    "consume", "absorbe", "necesito", "alto", "corto", "corta", "largo", "larga",
    "lujo", "premio", "reducido", "reducida", "apurado", "apurada",
    "presionado", "presionada", "poco", "extraño", "extraña",
    # -- ampliación v2/v3 --
    "flojera", "aburrido", "aburrida", "aburre", "aburro", "aburren", "aburrimiento",
    "agotador", "agotadora", "estresante", "estresantes",
    "acumulan", "acumula", "acumulado", "acumulada", "acumulados", "acumuladas",
    "acumulacion", "acumulación",
    "dificil", "difícil", "dificiles", "difíciles",
    "imposible", "imposibles", "injusto", "injustos",
    "problema", "problemas", "enredo",
    "complican", "complica", "complicado", "complicada", "complicad",
    "exigen", "exigencia", "exigente", "exigentes",
    "exageran", "exagera", "exagerado", "exagerada",
    "abusen", "abusan", "abuso",
    "nervios", "destruido", "destruida",
    "pesadisimo", "pesadísimo", "pesadisima", "pesadísima",
    "pesadisimos", "pesadísimos", "pesadisimas", "pesadísimas",
    "sacrificar", "sacrificio", "sacrificando",
    "triste", "mal",
}

CONTRASTE = re.compile(r"\b(pero|aunque|sin embargo|no obstante)\b")

CATEGORIAS = ("placentero", "desagradable", "mixto")


def limpiar_para_etiquetado(texto: str) -> str:
    """Normaliza el texto (minúsculas, sin puntuación) preservando tildes/ñ."""
    t = texto.lower()
    t = re.sub(r"[^\w\sáéíóúüñ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # "Bueno, ..." / "Bueno ..." al INICIO del texto casi siempre es una
    # muletilla discursiva ("Well, ..."), no una evaluación positiva. Se
    # retira solo esa ocurrencia inicial; un "bueno" en medio del texto
    # (p.ej. "me siento bueno y alegre") sí se conserva como evidencia.
    t = re.sub(r"^buen[oa]\s+", "", t)
    return t


def etiquetar(texto: str) -> dict:
    """
    Clasifica un texto en 'placentero', 'desagradable' o 'mixto'.

    Devuelve un diccionario con la etiqueta y la evidencia usada, para
    que el resultado sea auditable (no es una caja negra).
    """
    t = limpiar_para_etiquetado(texto)

    neg_score = 0
    pos_score = 0
    evidencia_neg = []
    evidencia_pos = []
    tramos_usados = []

    for patron in FRASES_NEGATIVAS:
        for m in re.finditer(patron, t):
            neg_score += 2
            evidencia_neg.append(m.group(0))
            tramos_usados.append((m.start(), m.end()))

    for patron in FRASES_POSITIVAS:
        for m in re.finditer(patron, t):
            pos_score += 2
            evidencia_pos.append(m.group(0))
            tramos_usados.append((m.start(), m.end()))

    for patron in FRASES_NEUTRALIZADORAS:
        for m in re.finditer(patron, t):
            tramos_usados.append((m.start(), m.end()))

    if tramos_usados:
        chars = list(t)
        for s, e in tramos_usados:
            for i in range(s, e):
                chars[i] = " "
        t_restante = "".join(chars)
    else:
        t_restante = t

    for tok in t_restante.split():
        if tok in POS_WORDS:
            pos_score += 1
            evidencia_pos.append(tok)
        if tok in NEG_WORDS:
            neg_score += 1
            evidencia_neg.append(tok)

    tiene_contraste = bool(CONTRASTE.search(t))

    if pos_score > 0 and neg_score > 0:
        etiqueta = "mixto"
    elif neg_score > pos_score:
        etiqueta = "desagradable"
    elif pos_score > neg_score:
        etiqueta = "placentero"
    else:
        etiqueta = "revisar_manualmente"

    return {
        "etiqueta": etiqueta,
        "pos_score": pos_score,
        "neg_score": neg_score,
        "contraste": tiene_contraste,
        "evidencia_pos": evidencia_pos,
        "evidencia_neg": evidencia_neg,
    }


def etiquetar_lote(textos):
    """Aplica etiquetar() a una lista/Serie de textos y devuelve una lista de dicts."""
    return [etiquetar(t) for t in textos]
