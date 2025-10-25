# Wykop API

https://github.com/lukas346/wykop_sdk_reloaded/tree/main

Sposób na znaleziska z przeszłości: wayback machine wykopaliska ALBO
szukanie w google tak: site:wykop.pl/link "znalezisko" after:2022-01-01 before:2022-02-01 (chociaż niekoniecznie będzie działać)

# Dane z labelami o nieprawdziwej informacji

Dane z oznaczonymi labelami zostały pozyskane ze strony https://wykoppl-informacjanieprawdziwa.surge.sh/ Przy użyciu pliku `scrapers/misinformation_scraper.py` w folderze `scrapers`

Stworzony został też scraper na wykopaliska: `scrapers/wykopalisko_scraper.py`, ale prawdopodobnie nie będzie użyteczny

## Instrukcja działania scrapera do danych z `wykopaliska`

`scrapers/scraper_config.json` musi zawierać poprawną ścieżkę do pliku wykonawczego firefox, np: `"firefox_path": "C:/Program Files/Mozilla Firefox/firefox.exe"`

## Instrukcja działania scrapera do danych z `głównej`

W folderze głównym projektu powinny znaleźć się dwa pliki:

- `klucz_do_api.txt`
- `secret_key.txt`

Oba można zdobyć poprzez stworzenie aplikacji na [wykopie dla programistów](https://dev.wykop.pl/) i odpowiednie przypisanie uzyskanych danych do tychże plików.

Praca następnie jest dzielona na dwa skrypty :

1. `scrapers/homepage_scraper.py` - pobranie wszystkich id postów z głównej za pomocą API. Pobiera 250 stron, po 40 id każda co oznacza 10_000 danych.

2. `scrapers/post_scraper.py` - pobranie wszystkich informacji z postów z pliku `scrapers/homepage_wykop_ids.json` który jest utworzony po zakończeniu działania pliku `scrapers/homepage_scraper.py`

## Instrukcja działania scrapera do danych po `id postu`

Posty mimo, że nie są dostępne na wykopalisku ani na głównej, to są często dostępne poprzez link: `https://wykop.pl/link/{id}`, gdzie `id` to numer postu.

Często posty są niedostępne i zwracają kod `404`, jednak możliwe jest uzyskanie archiwalnych postów i sprawdzenie każdego linku poprzez odwołanie do jego id.

`scrapers/id_range_posts.py` - Pobiera asychnronicznie przy użyciu zadanych `proxy` posty, sprawdza które posty są dostępne, które nie są i zapisuje niezbędne informacje.

Szczegóły działania:

1. Darmowe 10 adresów `proxy` można uzyskać poprzez rejestracje na `webshare` i skopiowanie ich do pliku `scrapers/proxies.txt`. Pozwala to na kilkukrotne przyspieszenie pobierania danych.
2. Plik zakłada podział pracy gdzie zapisuje po `n` postów do plików w folderze `posts`
3. Należy zadać odpowiednie parametry:

   - `batch_size` - po ile postów jest zapisywane w plikach `.jsonl`
   - `number_of_batches` - ile plików powstanie
   - `max_id` - maksymalne **id** postu podczas scrapowania

   Na podstawie powyższych zmiennych dopasowane są parametry scrapera

4. Asynchronicznie pobiera dane i na końcu merguje do pliku `data/final_combined_results.json`

## Spostrzeżenia:

- Pobieranie informacji o postach zajmuje około 2 posty/s
- Używanie scrapera zajmuje podobną ilość czasu standardowo, lecz można skalować o dodanie proxy (darmowe 10 dostępne przez `webshare`.)
