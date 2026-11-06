from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(11)


def h(text, level=1):
    return doc.add_heading(text, level=level)


def para(text, bold=False, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    return p


def bullets(items):
    for it in items:
        doc.add_paragraph(it, style='List Bullet')


def pic(path, caption, width=6.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(width))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = None if False else r.font.color.rgb


SHOTS = "screenshots/"

# Portada
title = doc.add_heading('UFC Fight Predictor', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run('Documentacion del proyecto - Aplicacion web de prediccion de peleas UFC')
r.italic = True
r.font.size = Pt(13)
doc.add_paragraph()
info = doc.add_paragraph()
info.alignment = WD_ALIGN_PARAGRAPH.CENTER
info.add_run('Proyecto academico - Version 1.0 - Noviembre 2026').font.size = Pt(10)
doc.add_page_break()

h('1. Resumen del proyecto', 1)
para('UFC Fight Predictor es una aplicacion web interactiva desarrollada en Python con '
     'Streamlit que permite analizar y predecir resultados de peleas de UFC (Mixed Martial Arts). '
     'El sistema combina dos modelos de Machine Learning entrenados con datos historicos de UFC '
     'para estimar: (a) que peleador tiene mayor probabilidad de ganar y (b) por que metodo '
     'podria producirse la victoria (KO/TKO, sumision o decision).')
para('La aplicacion tiene dos modos de uso: explorar el calendario real de eventos proximos '
     'obtenido en vivo desde una API externa, y crear una pelea personalizada seleccionando '
     'dos peleadores, una division y el contexto del evento.')
pic(SHOTS + "01_inicio.png",
    "Figura 1. Pantalla principal con las dos opciones de uso.")

h('2. Objetivos', 1)
bullets([
    'Demostrar la aplicacion practica de modelos predictivos de Machine Learning en un dominio real (deportes de combate).',
    'Integrar datos historicos de UFCStats y un dataset publico con cuotas de apuestas para entrenar modelos robustos.',
    'Ofrecer una interfaz web sencilla que permita comparar peleadores y obtener probabilidades estimadas.',
    'Incorporar informacion actualizada de eventos y peleadores mediante la API de UFCalendar.',
    'Evaluar honestamente el rendimiento del modelo usando una separacion cronologica de los datos (entrenar con peleas antiguas, probar con recientes).',
])

h('3. Funcionalidades de la aplicacion', 1)
h('3.1 Modo "Proximas peleas" (datos en vivo)', 2)
para('Al seleccionar esta opcion, la aplicacion consulta la API de UFCalendar y muestra los '
     'eventos de UFC programados en las proximas dos semanas. Para cada evento se muestra:')
bullets([
    'Nombre del evento y fecha.',
    'Sede/ubicacion (nombre del recinto y ciudad).',
    'Horarios de cada seccion de la tarjeta (prelims tempranos, prelims y main card) convertidos a hora de Colombia.',
    'La tarjeta completa de peleas en formato de cuadricula de dos columnas.',
])
para('Cada pelea muestra:')
bullets([
    'Los nombres de ambos peleadores.',
    'La division o peso (ej. Lightweight, Welterweight, Womens Flyweight).',
    'El record profesional de cada peleador en formato victorias-derrotas-empates (ej. 19-2-0), obtenido directamente de la API.',
    'La seccion de la tarjeta a la que pertenece (prelims o main card) con su hora aproximada en Colombia.',
    'Un indicador "Titulo" cuando la pelea es por el cinturon.',
    'Las cuotas de consenso de las casas de apuestas (formato americano) cuando la API las tiene disponibles.',
    'Un boton "Analizar" que lleva directamente a la pelea en modo personalizado con las cuotas ya cargadas.',
])
pic(SHOTS + "02_proximas_peleas.png",
    "Figura 2. Vista de proximas peleas: eventos, horarios en hora Colombia, division, records y seccion de cada combate.")
h('3.2 Modo "Pelea personalizada"', 2)
para('Permite al usuario crear una pelea a voluntad siguiendo un asistente de varios pasos:')
bullets([
    'Seleccion de peleador para la esquina roja y para la esquina azul (con buscador; no se permite elegir el mismo peleador dos veces).',
    'Definicion del contexto: division de peso, si es pelea por titulo (5 asaltos) y ubicacion del evento.',
    'Cuotas de apuestas americanas: si la pelea viene de "Proximas peleas" se cargan automaticamente desde la API (consenso de casas de apuestas); en una pelea personalizada se pueden ingresar manualmente (ej. -150, +130) o dejarlas vacias. Siempre son editables antes de analizar.',
    'Generacion del analisis predictivo.',
])
pic(SHOTS + "03_contexto_p_eleja.png",
    "Figura 3. Paso de contexto de la pelea: division, pelea por titulo y ubicacion.")
pic(SHOTS + "04_cuotas.png",
    "Figura 4. Paso opcional de cuotas de apuestas americanas.")
h('3.3 Resultados del analisis', 2)
bullets([
    'Probabilidad estimada de victoria para cada esquina, con barra de progreso y prediccion destacada.',
    'Probabilidades de metodo de victoria (KO/TKO, sumision, decision) desglosadas por resultado y agrupadas por metodo.',
    'Comparacion de estadisticas previas de ambos peleadores (edad, estatura, peso, rachas, efectividad de striking, derribos, etc.).',
])
pic(SHOTS + "05_resultados.png",
    "Figura 5. Resultados del analisis: probabilidad de ganador y metodos de victoria.")

h('4. Arquitectura y funcionamiento tecnico', 1)
h('4.1 Stack tecnologico', 2)
bullets([
    'Lenguaje: Python 3.',
    'Interfaz web: Streamlit (aplicacion local interactiva).',
    'Modelos de Machine Learning: RandomForest (scikit-learn) entrenados y serializados con joblib.',
    'Manipulacion de datos: pandas y NumPy.',
    'Consumo de APIs: requests.',
    'Manejo de zonas horarias: zoneinfo (America/Bogota).',
])
h('4.2 Flujo de datos', 2)
para('El flujo principal de la aplicacion es el siguiente:')
bullets([
    'La aplicacion carga dos modelos entrenados: el modelo de ganador y el modelo de esquina + metodo de victoria.',
    'Para una pelea personalizada, construye un vector de caracteristicas combinando el perfil historico mas reciente de cada peleador (estadisticas acumuladas hasta su ultima pelea registrada en el dataset) con variables del evento: division, si es por titulo, ubicacion, pais, numero de asaltos, genero y cuotas.',
    'Se calculan ademas variables de diferencia entre ambos peleadores (diferencia de edad, estatura, alcance, rachas, victorias, etc.) y el rango de cada uno dentro de su division.',
    'El modelo de ganador devuelve la probabilidad de que gane la esquina roja; la del azul es el complemento.',
    'El modelo de metodo de victoria devuelve probabilidades sobre seis resultados combinados (esquina + metodo) que se agrupan por metodo.',
])
h('4.3 Fuentes de datos', 2)
bullets([
    'UFCStats: estadisticas oficiales historicas de peleadores y eventos (datos crudos en data/raw/ufcstats/).',
    'Ultimate UFC Dataset (GitHub, shortlikeafox): conjunto publico que incluye cuotas historicas de apuestas; llega hasta el 28 de marzo de 2026.',
    'UFCalendar API (api.ufcalendar.com): eventos proximos, tarjetas de pelea, horarios por seccion y perfiles/records actualizados de peleadores en tiempo real.',
])
h('4.4 Preparacion de datos y entrenamiento', 2)
para('El proyecto incluye scripts de entrenamiento que:')
bullets([
    'Descargan y cruzan las fuentes de datos.',
    'Construyen perfiles historicos por peleador (la informacion usada para cada pelea corresponde unicamente a datos anteriores a esa pelea, evitando fugas de informacion).',
    'Calculan mas de cien variables predictivas por esquina y sus diferencias.',
    'Separan cronologicamente los datos: entrenamiento con peleas antiguas y prueba con las mas recientes.',
    'Entrenan y guardan los modelos en la carpeta models/.',
])

h('5. Resultados del modelo', 1)
para('Segun la evaluacion registrada en el proyecto (datos hasta el 28-03-2026):')
bullets([
    'Accuracy del modelo de ganador (con cuotas): 69.1%.',
    'Accuracy del modelo de esquina + metodo de victoria: 36.7% (experimental).',
])
para('Las probabilidades del modelo de metodo no estan calibradas y deben interpretarse con cautela.')

h('6. Limitaciones y consideraciones importantes', 1)
bullets([
    'La aplicacion es una demostracion historica/educativa, no un sistema de prediccion en vivo ni consejo de apuestas.',
    'El dataset de cuotas termina el 28 de marzo de 2026; no incluye informacion posterior.',
    'Los perfiles de peleadores se construyen con la aparicion mas reciente en los datos y pueden estar desactualizados (lesiones, cambios de equipo, etc.).',
    'Si se omiten las cuotas, el modelo imputa valores tipicos del entrenamiento; esto no equivale al modelo sin cuotas.',
    'Las probabilidades no estan calibradas y no garantizan resultados.',
    'La API de UFCalendar proporciona horarios por seccion de la tarjeta (prelims/main), no la hora exacta de cada pelea individual.',
])

h('7. Como ejecutar la aplicacion', 1)
para('Desde la carpeta ufc-predictor:')
doc.add_paragraph('python -m pip install -r requirements.txt', style='List Number')
doc.add_paragraph('python -m streamlit run app.py', style='List Number')
para('La aplicacion queda disponible en http://localhost:8501. Para usar el modo de proximas '
     'peleas se requiere la API key de UFCalendar configurada en el codigo (variable UFCAL_API_KEY).')

h('8. Estructura del proyecto', 1)
table = doc.add_table(rows=1, cols=2)
table.style = 'Light Grid Accent 1'
hdr = table.rows[0].cells
hdr[0].text = 'Carpeta / Archivo'
hdr[1].text = 'Descripcion'
rows = [
    ('app.py', 'Aplicacion web Streamlit: interfaz, integracion con APIs y visualizacion de resultados.'),
    ('src/train_updated_model.py', 'Prepara y evalua el conjunto de caracteristicas historicas de UFCStats.'),
    ('src/train_odds_model.py', 'Entrena los modelos con cuotas y el modelo de esquina + metodo.'),
    ('src/train_and_report.py', 'Scripts auxiliares de entrenamiento y generacion de reportes.'),
    ('data/raw/', 'Copias originales de las fuentes de datos. No modificar manualmente.'),
    ('data/processed/', 'Tablas de caracteristicas procesadas para entrenamiento y perfiles.'),
    ('models/', 'Modelos RandomForest entrenados (.joblib).'),
    ('reports/', 'Metricas, predicciones, informes y graficas generadas.'),
    ('notebooks/', 'Notebooks de exploracion y resultados del proyecto.'),
]
for a, b in rows:
    cells = table.add_row().cells
    cells[0].text = a
    cells[1].text = b

doc.add_paragraph()
para('Documento generado como parte del proyecto UFC Fight Predictor.', italic=True)

out = 'Documentacion_UFC_Fight_Predictor.docx'
doc.save(out)
print('Guardado:', out)
