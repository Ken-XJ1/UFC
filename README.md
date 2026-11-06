# Predictor de peleas de UFC

Proyecto en desarrollo para analizar peleas históricas y estimar resultados UFC. El flujo principal usa estadísticas previas al combate y cuotas históricas; se evalúa con una separación cronológica para entrenar con peleas antiguas y probar con peleas posteriores.

## Estructura actual

- `data/raw/`: copias de las fuentes originales. No modificar manualmente.
- `data/processed/`: tablas preparadas para entrenar y validar.
- `notebooks/`: notebooks de las etapas de exploración y resultados. Se consolidarán en un solo notebook cuando se cierre el proyecto.
- `src/train_updated_model.py`: prepara y evalúa el conjunto ampliado de UFCStats.
- `src/train_odds_model.py`: entrena los modelos con cuotas y un modelo adicional de esquina y método de victoria.
- `models/`: modelos entrenados guardados.
- `reports/`: métricas, predicciones, informes y gráficas generados.

## Preparar los datos y entrenar

Instala primero las dependencias de entrenamiento y ejecuta estos comandos desde la carpeta `ufc-predictor`:

```powershell
python -m pip install -r requirements-train.txt
python src\train_updated_model.py --refresh
python src\train_odds_model.py --refresh
```

El primer comando actualiza los CSV de UFCStats y prepara las características históricas necesarias para auditar etiquetas. El segundo descarga el conjunto público [Ultimate UFC Dataset](https://github.com/shortlikeafox/ultimate_ufc_dataset), comprueba los resultados coincidentes frente a UFCStats y entrena los modelos. Las actualizaciones son manuales; no se ejecutan automáticamente cuando cambian las fuentes.

Si ya se descargaron los CSV y solo quieres reentrenar:

```powershell
python src\train_updated_model.py
python src\train_odds_model.py
```

El archivo original `data/raw/data.csv` se conserva intacto. Los CSV de UFCStats se guardan aparte en `data/raw/ufcstats/`; el conjunto con cuotas, en `data/raw/external/`.

## Resultados y notebook

Abre `notebooks/02_resultados_ufcstats_actualizados.ipynb` para ver el informe narrativo en orden de presentación: fuentes, método, comparación de modelos, evaluación frente al favorito en las mismas peleas y límites. El experimento con cuotas está separado de la primera versión del proyecto.

## Aplicación web de demostración

La aplicación Streamlit ofrece dos modos:

- **Elegir peleadores:** busca y selecciona por separado un atleta para cada esquina, define la división y el contexto, y obtiene una predicción de los modelos guardados. Las cuotas americanas pueden introducirse de forma opcional.
- **Explorar pelea histórica:** selecciona un combate del conjunto de prueba, compara la predicción con las estadísticas y revela el resultado real registrado.

Iníciala desde esta carpeta:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

La aplicación es una demo local; para que el profesor la abra en su propio equipo, se necesita compartir o alojar el proyecto y sus datos/modelos. No es una predicción en vivo: el dataset de cuotas llega hasta el 28-03-2026. En el modo personalizado, cada perfil usa la aparición más reciente del peleador en los datos y puede estar desactualizado. No representa noticias, lesiones o cambios posteriores. El modelo de ganador fue entrenado con cuotas; al omitirlas, la aplicación utiliza la imputación de valores típicos del entrenamiento, lo que no equivale al modelo sin cuotas. Las probabilidades de método son experimentales y no están calibradas.

Los resultados con cuotas se guardan en:

- `reports/metricas_modelos_con_cuotas.csv`
- `reports/predicciones_prueba_con_cuotas.csv`
- `reports/Informe_UFC_con_cuotas.docx`
- `reports/figures/17_comparacion_modelos_con_cuotas.png`
- `models/random_forest_ufc_odds.joblib`

El modelo adicional que estima uno de seis resultados (roja/azul por KO/TKO, sumisión o decisión) guarda:

- `reports/metricas_metodo_victoria.csv`
- `reports/predicciones_metodo_victoria.csv`
- `reports/classification_report_metodo_victoria.txt`
- `reports/figures/19_matriz_confusion_metodo_victoria.png`
- `models/random_forest_ufc_method.joblib`

En la ejecución actual, el Random Forest que predice esquina y método obtuvo 36,7 % de accuracy y 0,355 de macro F1 en 1.486 peleas de prueba (29-04-2023 a 28-03-2026). Ese problema tiene seis clases; no es directamente comparable con el modelo binario de ganador. Se excluyeron 258 filas con método desconocido u otros resultados como descalificación, no contest o resultado anulado.

## Cómo interpretar las métricas

En el experimento binario con cuotas, el Random Forest obtuvo 69,1 % de accuracy y el modelo equivalente sin cuotas, 63,5 % en las mismas 1.499 peleas de prueba (22-04-2023 a 28-03-2026). Para comparar con el favorito del mercado, se volvieron a evaluar ambos métodos en las mismas 1.275 peleas que tenían cuotas para ambas esquinas: Random Forest acertó 71,1 % y el favorito 69,6 %. La ventaja observada del modelo fue de 57 peleas frente a 38 en sentido contrario; la prueba pareada exacta dio p = 0,064. Esta diferencia no basta para afirmar que el modelo sea superior al mercado.

Las cuotas resumen la expectativa del mercado: el modelo con cuotas no predice usando solo estadísticas deportivas. Las predicciones históricas no garantizan resultados futuros ni son consejo de apuestas. Las probabilidades de método son estimaciones experimentales y deben interpretarse con cautela.

La fuente de cuotas se publica con licencia CC BY 4.0; acredita a Matt Dabbert y a los colaboradores si reutilizas esos datos. Los CSV de UFCStats provienen de `Greco1899/scrape_ufc_stats`, cuyo código se publica bajo GPL-3.0.
