# Wykop API

https://github.com/lukas346/wykop_sdk_reloaded/tree/main

Sposób na znaleziska z przeszłości: wayback machine wykopaliska ALBO
szukanie w google tak: site:wykop.pl/link "znalezisko" after:2022-01-01 before:2022-02-01 (chociaż niekoniecznie będzie działać)

# Dane z labelami o nieprawdziwej informacji

Dane z oznaczonymi labelami zostały pozyskane ze strony https://wykoppl-informacjanieprawdziwa.surge.sh/ Przy użyciu pliku `misinformation_scraper.py` w folderze `scrapers`

Stworzony został też scraper na wykopaliska: `wykopalisko_scraper.py`, ale prawdopodobnie nie będzie użyteczny

## Instrukcja działania scrapera do danych z wykopaliska

`scraper_config.json` musi zawierać poprawną ścieżkę do pliku wykonawczego firefox, np: `"firefox_path": "C:/Program Files/Mozilla Firefox/firefox.exe"`
