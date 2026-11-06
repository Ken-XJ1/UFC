import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
OUT = "screenshots/06_ejemplo_nolan_moicano.png"


def pick_fighter(page, combobox_label, name):
    # Streamlit selectbox: clic en el combobox y escribir para filtrar
    box = page.get_by_role("combobox").nth(
        0 if "roja" in combobox_label.lower() or combobox_label == "rojo" else 1
    )
    box.click()
    page.wait_for_timeout(500)
    box.fill(name)
    page.wait_for_timeout(800)
    page.keyboard.press("Enter")
    page.wait_for_timeout(800)


with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page(viewport={"width": 1400, "height": 1000})
    pg.goto(URL, wait_until="networkidle")
    pg.wait_for_timeout(3000)

    pg.get_by_role("button", name="Crear mi propia pelea").click()
    pg.wait_for_timeout(4000)

    boxes = pg.get_by_role("combobox")
    print("comboboxes:", boxes.count())

    # Esquina roja: Tom Nolan ; azul: Renato Moicano
    boxes.nth(0).click()
    pg.wait_for_timeout(500)
    pg.keyboard.type("Tom Nolan")
    pg.wait_for_timeout(1000)
    pg.keyboard.press("Enter")
    pg.wait_for_timeout(1000)

    boxes.nth(1).click()
    pg.wait_for_timeout(500)
    pg.keyboard.type("Renato Moicano")
    pg.wait_for_timeout(1000)
    pg.keyboard.press("Enter")
    pg.wait_for_timeout(1000)

    pg.screenshot(path="screenshots/debug_select.png", full_page=True)
    print(pg.locator("body").inner_text()[-1200:])

    pg.get_by_role("button", name="Siguiente").click()
    pg.wait_for_timeout(3000)

    # Paso contexto: defaults ya son Lightweight / Las Vegas / sin titulo
    pg.get_by_role("button", name="Siguiente").click()
    pg.wait_for_timeout(3000)

    # Paso cuotas
    inputs = pg.locator("input[type='text']")
    print("text inputs:", inputs.count())
    # Identificar por placeholder
    pg.get_by_placeholder("-150 o +130").fill("-111")
    pg.get_by_placeholder("+130 o -150").fill("-115")
    pg.wait_for_timeout(500)

    pg.get_by_role("button", name="Comenzar análisis").click()
    pg.wait_for_timeout(10000)

    pg.screenshot(path=OUT, full_page=True)
    txt = pg.locator("body").inner_text()
    # guardar resultado en archivo para revisar
    open("screenshots/resultado_nolan_moicano.txt", "w", encoding="utf-8").write(txt)
    print(txt[-4000:])
    b.close()
