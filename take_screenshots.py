import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path("screenshots")
OUT.mkdir(exist_ok=True)
URL = "http://localhost:8501"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(URL, wait_until="networkidle")
    page.wait_for_timeout(3000)
    page.screenshot(path=str(OUT / "01_inicio.png"), full_page=True)

    # Paso 2: proximas peleas
    page.get_by_role("button", name="Ver próximas peleas").click()
    page.wait_for_timeout(8000)  # cargar eventos + cards + records
    page.screenshot(path=str(OUT / "02_proximas_peleas.png"), full_page=True)

    # Click en Analizar de la primera pelea -> va a paso 4
    btn = page.get_by_role("button", name="Analizar").first
    if btn.count():
        btn.click()
        page.wait_for_timeout(4000)
        page.screenshot(path=str(OUT / "03_contexto_p_eleja.png"), full_page=True)
        # Siguiente (division/ubicacion ya tienen defaults)
        page.get_by_role("button", name="Siguiente").click()
        page.wait_for_timeout(3000)
        page.screenshot(path=str(OUT / "04_cuotas.png"), full_page=True)
        page.get_by_role("button", name="Comenzar análisis").click()
        page.wait_for_timeout(8000)
        page.screenshot(path=str(OUT / "05_resultados.png"), full_page=True)
    else:
        print("No se encontro boton Analizar")

    browser.close()
print("Screenshots en", OUT)
