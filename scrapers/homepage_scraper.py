### Homepage scraper - zbieranie id linków z homepage Wykopu

import json
from wykop_sdk_reloaded.v3.client import AuthClient, WykopApiClient
from wykop_sdk_reloaded.v3.types import LinkType
import datetime
from tqdm import tqdm


def dump_json(obj, fname):
    with open(fname, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=2)


# --- Autoryzacja aplikacji ---
app_key = open("klucz_do_api.txt").read().strip()
app_secret = open("secret_key.txt").read().strip()

auth = AuthClient()
auth.authenticate_app(app_key, app_secret)
api = WykopApiClient(auth)

# --- Zbieranie danych ---
all_ids = []

# Przykład: iteracja po kilku stronach głównej
for page in tqdm(
    range(1, 250), desc="Pobieranie stron z Homepage wykop.pl"
):  # 10_000 max, gdzie każda strona to 40 linków := 250 stron
    # print(f"Pobieram stronę {page}...")
    links_page = api.links_list_links(type=LinkType.HOMEPAGE, page=page)
    all_ids.extend([link["id"] for link in links_page["data"]])

# --- Zapis dopiero na końcu ---
merged = {"ids": list(set(all_ids))}

dump_json(merged, "homepage_wykop_ids.json")

print(f"Zapisano {len(all_ids)} id postów do homepage_wykop_ids.json")
