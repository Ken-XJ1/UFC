from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
SCREENSHOTS = ROOT / "screenshots"
OUTPUT = ROOT / "Documentacion_UFC_Fight_Predictor_mejorada.docx"

doc = Document()
section = doc.sections[0]
section.top_margin = Inches(0.7)
section.bottom_margin = Inches(0.7)
section.left_margin = Inches(0.85)
section.right_margin = Inches(0.85)

normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(10.5)
normal.paragraph_format.space_after = Pt(5)
for name, size, color in (
    ("Title", 30, "172554"),
    ("Heading 1", 19, "172554"),
    ("Heading 2", 13, "1D4ED8"),
    ("Heading 3", 11, "1E3A8A"),
):
    style = doc.styles[name]
    style.font.name = "Calibri"
    style.font.size = Pt(size)
    style.font.bold = True
    style.font.color.rgb = RGBColor.from_string(color)

header = section.header.paragraphs[0]
header.text = "UFC FIGHT PREDICTOR  |  DOCUMENTACIÓN DEL PROYECTO"
header.style = doc.styles["Caption"]
header.alignment = WD_ALIGN_PARAGRAPH.RIGHT

footer = section.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
footer.add_run("Proyecto académico · Noviembre de 2026  |  Página ")
field = OxmlElement("w:fldSimple")
field.set(qn("w:instr"), "PAGE")
footer._p.append(field)


def shade(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def set_cell_text(cell, text: str, *, bold: bool = False, color: str | None = None) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(1)
    run = paragraph.add_run(text)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def heading(text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def paragraph(text: str = "", *, bold: bool = False, italic: bool = False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    return p


def bullet(text: str) -> None:
    doc.add_paragraph(text, style="List Bullet")


def numbered(text: str) -> None:
    doc.add_paragraph(text, style="List Number")


def add_link(paragraph_obj, label: str, url: str) -> None:
    relationship_id = paragraph_obj.part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship_id)
    run = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    properties.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    properties.append(underline)
    run.append(properties)
    text = OxmlElement("w:t")
    text.text = label
    run.append(text)
    hyperlink.append(run)
    paragraph_obj._p.append(hyperlink)


def figure(filename: str, caption: str, width: float = 6.1) -> None:
    path = SCREENSHOTS / filename
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró la captura {path}")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Inches(width))
    caption_paragraph = doc.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_paragraph.paragraph_format.space_after = Pt(8)
    caption_run = caption_paragraph.add_run(caption)
    caption_run.italic = True
    caption_run.font.size = Pt(9)
    caption_run.font.color.rgb = RGBColor.from_string("475569")


def make_table(headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Shading Accent 1"
    for cell, text in zip(table.rows[0].cells, headers):
        set_cell_text(cell, text, bold=True, color="FFFFFF")
        shade(cell, "1D4ED8")
    for row_index, values in enumerate(rows):
        cells = table.add_row().cells
        for cell, text in zip(cells, values):
            set_cell_text(cell, text)
            if row_index % 2:
                shade(cell, "EFF6FF")
    doc.add_paragraph()


# Portada
doc.add_paragraph()
doc.add_paragraph()
title = doc.add_paragraph(style="Title")
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.add_run("UFC Fight Predictor")
subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle_run = subtitle.add_run(
    "Documentación del proyecto y guía de la aplicación"
)
subtitle_run.font.size = Pt(15)
subtitle_run.font.color.rgb = RGBColor.from_string("475569")
doc.add_paragraph()
summary = doc.add_paragraph()
summary.alignment = WD_ALIGN_PARAGRAPH.CENTER
summary.add_run(
    "Proyecto académico de ciencia de datos\n"
    "Aplicación interactiva para explorar predicciones de peleas de UFC\n"
    "Versión documental: noviembre de 2026"
)
doc.add_paragraph()
cover_note = doc.add_paragraph()
cover_note.alignment = WD_ALIGN_PARAGRAPH.CENTER
note_run = cover_note.add_run(
    "Las predicciones son estimaciones experimentales; no garantizan resultados "
    "ni constituyen consejo de apuestas."
)
note_run.italic = True
note_run.font.color.rgb = RGBColor.from_string("9A3412")
doc.add_page_break()

heading("1. Resumen del proyecto")
paragraph(
    "UFC Fight Predictor es una aplicación web académica desarrollada en Python "
    "con Streamlit. Permite explorar dos tareas de clasificación: estimar qué "
    "esquina podría ganar y estimar una de seis combinaciones de esquina y método "
    "(KO/TKO, sumisión o decisión). La aplicación utiliza modelos Random Forest "
    "entrenados con datos históricos."
)
paragraph(
    "La interfaz tiene dos recorridos: consultar eventos próximos mediante "
    "UFCalendar, cuando la API está disponible, o crear una pelea personalizada "
    "eligiendo a los dos atletas y el contexto. La información del calendario "
    "puede ser actual; eso no significa que las estadísticas usadas por los "
    "modelos también estén actualizadas."
)
figure("01_inicio.png", "Figura 1. Pantalla inicial y opciones de la aplicación.")

heading("2. Objetivos")
for text in [
    "Aplicar y explicar modelos de clasificación a un problema deportivo real.",
    "Preparar estadísticas históricas y cuotas de peleas para crear variables predictivas.",
    "Evaluar los modelos con una separación cronológica: entrenar con peleas anteriores y probar con peleas posteriores.",
    "Presentar las estimaciones en una interfaz interactiva y explicar sus limitaciones.",
]:
    bullet(text)

heading("3. Funcionalidades de la aplicación")
heading("3.1 Eventos próximos", 2)
paragraph(
    "La aplicación consulta UFCalendar para mostrar eventos programados, tarjetas "
    "de peleas y datos de evento que la API tenga disponibles. Puede presentar "
    "división, récords, sede, horarios generales por sección de la tarjeta y "
    "cuotas de consenso si la respuesta de la API las incluye."
)
paragraph(
    "Los horarios se convierten a la hora de Colombia y corresponden a secciones "
    "como preliminares o cartelera principal; no son necesariamente la hora exacta "
    "de inicio de cada combate. La disponibilidad depende de la API, la conexión "
    "y una clave válida."
)
figure(
    "02_proximas_peleas.png",
    "Figura 2. Ejemplo de evento y tarjeta consultados mediante la API.",
)

heading("3.2 Pelea personalizada", 2)
for text in [
    "Elegir por separado un peleador para la esquina roja y otro para la azul, con búsqueda de nombres y prevención de selecciones duplicadas.",
    "Definir la división, indicar si es una pelea por título y seleccionar la ubicación.",
    "Ingresar cuotas americanas manualmente o dejarlas vacías.",
    "Consultar probabilidades estimadas y estadísticas disponibles para ambos atletas.",
]:
    bullet(text)
figure(
    "03_contexto_p_eleja.png",
    "Figura 3. Selección de la división, la condición de título y la ubicación.",
)
figure(
    "04_cuotas.png",
    "Figura 4. Entrada opcional de cuotas americanas.",
)

heading("3.3 Resultados")
paragraph(
    "La pantalla de resultados muestra la probabilidad estimada para cada "
    "esquina y las seis probabilidades del modelo de método, que también se "
    "agrupan en KO/TKO, sumisión y decisión. Incluye una comparación de "
    "estadísticas históricas disponibles. En el recorrido de peleas históricas "
    "se puede contrastar la predicción con el resultado registrado; una pelea "
    "personalizada no tiene un resultado real conocido que mostrar."
)
figure("05_resultados.png", "Figura 5. Ejemplo de resultados de una pelea.")

heading("4. Datos y funcionamiento")
heading("4.1 Fuentes de datos", 2)
make_table(
    ["Fuente", "Uso en el proyecto", "Alcance y cuidado"],
    [
        [
            "UFCStats",
            "Resultados históricos y estadísticas de peleas.",
            "Los registros se procesan para construir variables previas al combate.",
        ],
        [
            "Ultimate UFC Dataset",
            "Datos históricos y cuotas utilizados en el experimento con mercado.",
            "La versión analizada llega hasta el 28 de marzo de 2026.",
        ],
        [
            "UFCalendar API",
            "Eventos próximos, tarjetas y datos de eventos disponibles.",
            "Es una fuente externa; la disponibilidad depende de la API y su clave.",
        ],
    ],
)

heading("4.2 Variables y prevención de fuga de información", 2)
paragraph(
    "En la evaluación histórica, las estadísticas de cada atleta se construyen "
    "con información anterior a la pelea que se intenta predecir. Las variables "
    "incluyen, entre otras, edad, estatura, alcance, récord y estadísticas "
    "acumuladas. También se calculan diferencias entre las esquinas. Las "
    "variables posteriores al combate se excluyen del entrenamiento para reducir "
    "la fuga de información."
)
paragraph(
    "En cada combate, roja y azul identifican las esquinas asignadas a ese "
    "enfrentamiento; no son una característica permanente del atleta. En una "
    "pelea personalizada, la app utiliza el perfil más reciente que encuentre "
    "en el conjunto histórico del proyecto. Ese perfil puede estar desactualizado "
    "y no refleja necesariamente lesiones, cambios de forma u otros hechos "
    "posteriores."
)
paragraph(
    "La aplicación consulta el calendario de forma separada: que una pelea o "
    "cuota aparezca en la API no actualiza automáticamente las estadísticas "
    "históricas del modelo."
)

heading("4.3 Modelos y evaluación cronológica", 2)
paragraph(
    "El script de entrenamiento compara regresión logística, Random Forest y "
    "XGBoost para predecir la esquina ganadora. La aplicación carga el Random "
    "Forest guardado. Para el problema de esquina y método se utiliza otro "
    "Random Forest con seis clases. La separación de entrenamiento y prueba se "
    "hace por fecha, no de forma aleatoria."
)

heading("5. Resultados y cómo interpretarlos")
paragraph(
    "Las cifras siguientes corresponden a los archivos de evaluación generados "
    "por el proyecto. Accuracy es la proporción de ejemplos de prueba clasificados "
    "correctamente; no es una garantía ni una probabilidad de acierto para una "
    "pelea concreta."
)
make_table(
    ["Experimento", "Resultado", "Muestra de prueba"],
    [
        [
            "Ganador, Random Forest con cuotas",
            "69,1 % accuracy",
            "1.499 peleas; 22-04-2023 a 28-03-2026",
        ],
        [
            "Ganador, Random Forest sin cuotas",
            "63,5 % accuracy",
            "Las mismas 1.499 peleas",
        ],
        [
            "Ganador vs. favorito del mercado",
            "71,1 % el modelo; 69,6 % el favorito",
            "1.275 peleas con cuotas para ambas esquinas",
        ],
        [
            "Esquina + método, seis clases",
            "36,7 % accuracy; macro F1 de 0,355",
            "1.486 peleas; 29-04-2023 a 28-03-2026",
        ],
    ],
)
paragraph(
    "La comparación con el favorito se hizo sobre las mismas 1.275 peleas: el "
    "modelo acertó en 57 casos en que el favorito falló, mientras que el favorito "
    "acertó en 38 casos en que el modelo falló. La prueba pareada exacta dio "
    "p = 0,064. Como ese valor no alcanza el umbral convencional de 0,05, estos "
    "resultados no bastan para afirmar que el modelo supere al mercado. La "
    "diferencia observada tampoco demuestra que añadir cuotas cause por sí solo "
    "una mejora generalizable."
)
paragraph(
    "El problema de método tiene seis clases y es más difícil que elegir entre "
    "dos esquinas; su accuracy no debe compararse directamente con el problema "
    "binario. Se excluyeron 258 filas cuyo resultado no correspondía a las seis "
    "clases definidas, como métodos desconocidos, descalificaciones o no contest."
)
paragraph(
    "Las cuotas incorporan la expectativa del mercado, además de las estadísticas "
    "deportivas. Si no se proporcionan en una pelea personalizada, el pipeline "
    "rellena los valores faltantes con estadísticas típicas del entrenamiento; "
    "esto no es equivalente a usar el modelo entrenado sin cuotas. Las "
    "probabilidades de método son experimentales y no están calibradas."
)

heading("6. Limitaciones y uso responsable")
for text in [
    "Los datos históricos con cuotas llegan hasta el 28 de marzo de 2026; no incluyen automáticamente información posterior.",
    "Los perfiles personalizados pueden estar desactualizados y no incluyen necesariamente lesiones ni cambios recientes.",
    "El calendario y las cuotas de UFCalendar dependen de la API y no convierten al modelo en un servicio de predicción en vivo.",
    "Las métricas reflejan una muestra histórica de prueba; el rendimiento futuro puede diferir.",
    "Las estimaciones no garantizan resultados y no constituyen consejo de apuestas.",
    "Las cuotas faltantes se imputan en el modelo entrenado con cuotas; esa salida debe interpretarse con cautela.",
]:
    bullet(text)

heading("7. Ejecución")
paragraph("Desde la carpeta del proyecto, instala las dependencias e inicia Streamlit:")
numbered("python -m pip install -r requirements.txt")
numbered("python -m streamlit run app.py")
paragraph(
    "La aplicación local se abre normalmente en http://localhost:8501. Para "
    "consultar eventos próximos se necesita una clave válida de UFCalendar y "
    "acceso a Internet. Antes de compartir el proyecto, no dejes la clave escrita "
    "en el código ni en el documento: muévela a una configuración local de "
    "secretos o a una variable de entorno, excluida del control de versiones. "
    "Si una clave ya quedó incluida en una copia compartida, revócala y genera "
    "otra antes de distribuir el proyecto."
)

heading("8. Estructura principal del proyecto")
make_table(
    ["Ruta", "Descripción"],
    [
        ["app.py", "Aplicación web Streamlit."],
        ["src/train_updated_model.py", "Preparación y evaluación del conjunto histórico UFCStats."],
        ["src/train_odds_model.py", "Entrenamiento y evaluación de modelos con cuotas."],
        ["data/raw/", "Fuentes originales conservadas."],
        ["data/processed/", "Datos preparados y variables de entrada."],
        ["models/", "Modelos serializados que carga la aplicación."],
        ["reports/", "Métricas, predicciones e informes de evaluación."],
        ["notebooks/", "Análisis y documentación narrativa del proyecto."],
    ],
)

heading("9. Ejemplo práctico: Tom Nolan vs. Renato Moicano")
paragraph(
    "Como ejemplo de uso del modo personalizado se procesó el combate estelar de "
    "UFC Fight Night del 31 de octubre de 2026 en el Meta Apex de Las Vegas "
    "(Ligero, sin pelea por título). Las cuotas ingresadas fueron las del "
    "mercado: -111 para Tom Nolan y -115 para Renato Moicano, ambas en formato "
    "americano. Los perfiles de ambos atletas corresponden a su aparición más "
    "reciente en el conjunto histórico (27-09-2025 y 28-06-2025, respectivamente), "
    "por lo que pueden no reflejar su estado actual."
)
make_table(
    ["Salida del modelo", "Valor estimado"],
    [
        ["Probabilidad de victoria — Tom Nolan (rojo)", "56,0 %"],
        ["Probabilidad de victoria — Renato Moicano (azul)", "44,0 %"],
        ["Predicción del modelo", "Gana Tom Nolan"],
        ["Resultado combinado más probable", "Esquina azul — Decisión (22,6 % según el modelo de método)"],
    ],
)
paragraph(
    "El modelo de método estima las seis combinaciones posibles de esquina y "
    "forma de terminar (KO/TKO, sumisión o decisión) y las agrupa por método. "
    "En este ejemplo el resultado individual más probable fue una decisión a "
    "favor de la esquina azul, aunque el modelo de ganador favorece a Nolan. "
    "Esta diferencia ilustra que ambos modelos son independientes y que sus "
    "probabilidades no están calibradas ni garantizan el resultado real."
)
figure(
    "06_ejemplo_nolan_moicano.png",
    "Figura 6. Ejemplo real: probabilidad de ganador y métodos de victoria para Nolan vs. Moicano.",
    width=6.4,
)

heading("10. Fuentes y referencias")
references = [
    (
        "UFCStats",
        "https://ufcstats.com/",
        "Fuente de resultados y estadísticas históricas de UFC.",
    ),
    (
        "Scrape UFC Stats — Greco1899",
        "https://github.com/Greco1899/scrape_ufc_stats",
        "Repositorio utilizado para recopilar archivos de UFCStats; el código se publica bajo GPL-3.0.",
    ),
    (
        "Ultimate UFC Dataset — Matt Dabbert y colaboradores",
        "https://github.com/shortlikeafox/ultimate_ufc_dataset",
        "Conjunto público con cuotas históricas, publicado bajo CC BY 4.0. Debe conservarse la atribución al reutilizarlo.",
    ),
    (
        "UFCalendar",
        "https://ufcalendar.com/",
        "Servicio externo usado para consultar eventos y datos disponibles mediante API.",
    ),
]
for label, url, description in references:
    p = doc.add_paragraph(style="List Bullet")
    add_link(p, label, url)
    p.add_run(f" — {description}")

doc.core_properties.title = "UFC Fight Predictor — Documentación mejorada"
doc.core_properties.subject = (
    "Descripción de la aplicación, metodología, resultados y limitaciones"
)
doc.core_properties.author = "Proyecto UFC Fight Predictor"
doc.core_properties.keywords = (
    "UFC, ciencia de datos, aprendizaje automático, documentación"
)
doc.save(OUTPUT)
print(f"Documento guardado: {OUTPUT}")
