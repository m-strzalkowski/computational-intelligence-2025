from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
import random
import pandas as pd
import datetime
import os


firefox_options = Options()
firefox_options.headless = True

# Załaduj ścieżkę z pliku konfiguracyjnego
script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, "scraper_config.json"), "r") as f:
    import json

    config = json.load(f)
firefox_options.binary_location = config.get("firefox_path", None)
if not firefox_options.binary_location or not os.path.exists(
    firefox_options.binary_location
):
    raise ValueError(
        "Ścieżka do przeglądarki Firefox jest nieprawidłowa lub nie istnieje. Sprawdź plik scraper_config.json"
    )

# ------------------------
# Uruchomienie przeglądarki
# ------------------------
driver = webdriver.Firefox(options=firefox_options)

# ------------------------
# Pobranie linków z Wykopalisko
# ------------------------

base_url = "https://wykop.pl/wykopalisko"

driver.get(base_url)

# Pobierz liczbę stron
try:
    last_page_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "li.paging.last a"))
    )
    href = last_page_element.get_attribute("href")  # np. /wykopalisko/strona/7
    max_pages = int(href.split("/")[-1])

    # max_pages = int(last_page_element.text)
    print(f"Liczba stron: {max_pages}")
except Exception as e:
    print("Nie udało się pobrać liczby stron:", e)


# Poczekaj na pojawienie się przycisku "Akceptuj wszystkie" i kliknij go
try:
    accept_all_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable(
            (
                By.XPATH,
                "/html/body/div[2]/div[1]/div[2]/div[2]/div[2]/button[2]/div/b/span",
            )
        )
    )
    accept_all_button.click()
    print("Kliknięto przycisk 'Akceptuj wszystkie'.")
except Exception as e:
    print("Nie udało się kliknąć przycisku 'Akceptuj wszystkie':", e)

post_links = set()


# Pętla po stronach
for page in range(1, max_pages + 1):
    print(f"Przetwarzanie strony {page}/{max_pages}")
    driver.get(f"{base_url}/strona/{page}")

    # Poczekaj aż pojawią się linki
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located(
            (
                By.CSS_SELECTOR,
                'a[href^="/link/"]:not(section.sidebar a):not(div.sidebar a)',
            )
        )
    )
    # Scroll to the bottom of the page to load all items
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # pobierz href od razu, nie trzymaj WebElement-ów
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located(
                (
                    By.CSS_SELECTOR,
                    'a[href^="/link/"]:not(section.sidebar a):not(div.sidebar a)',
                )
            )
        )

        local_post_links = [
            link.get_attribute("href")
            for link in driver.find_elements(
                By.CSS_SELECTOR,
                'a[href^="/link/"]:not(section.sidebar a):not(div.sidebar a)',
            )
            if link.get_attribute("href")
        ]

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 2/3);")
        time.sleep(1.5)  # poczekaj na załadowanie nowych elementów
        old_count = len(post_links)
        post_links.update(local_post_links)

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height and len(post_links) == old_count:
            break
        last_height = new_height
    print(f"Znaleziono {len(local_post_links)} linków na {page} stronie")

print(f"\nZnaleziono {len(post_links)} linków Na wszystkich stronach")


data = []

for idx, link in enumerate(post_links):
    print(f"Pobieram post {idx+1}/{len(post_links)}: {link}")
    driver.get(link)

    # Poczekaj aż załaduje się cokolwiek
    try:
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div.body, section.info, section.blue.info")
            )
        )
    except:
        pass

    # Tytuł
    try:
        title = driver.find_element(By.CSS_SELECTOR, "h1.heading").text
    except:
        try:
            title = driver.find_element(
                By.CSS_SELECTOR, "div.body, section.info:not(.blue)"
            ).text.split("\n")[0]
        except:
            title = "Brak tytułu"

    # ALERT
    try:
        alert_section = driver.find_element(By.CSS_SELECTOR, "section.blue.info")
        alert_text = alert_section.text.strip()
    except:
        alert_text = ""

    # CONTENT (bez alertu)
    try:
        content_section = driver.find_element(
            By.CSS_SELECTOR, "div.body, section.info:not(.blue)"
        )
        content = content_section.text.strip()
    except:
        content = ""

    # TAGI
    try:
        tags = [t.text for t in driver.find_elements(By.CSS_SELECTOR, "li.tag a")]
    except:
        tags = []

    # DATA
    try:
        date = driver.find_element(By.CSS_SELECTOR, "time").get_attribute("datetime")
    except:
        date = None

    data.append(
        {
            "title": title,
            "link": link,
            "alert": alert_text,
            "content": content,
            "tags": tags,
            "date": date,
        }
    )

    # Minimalny random delay, żeby nie obciążać serwera
    time.sleep(random.uniform(0.3, 0.7))

# ------------------------
# Zamknięcie przeglądarki
# ------------------------
driver.quit()

# ------------------------
# Zapis do CSV
# ------------------------
df = pd.DataFrame(data)
date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")

save_dir = os.path.join("..", "data", "csv")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

df.to_csv(os.path.join(save_dir, f"wykopalisko_posts_{date_str}.csv"), index=False)
