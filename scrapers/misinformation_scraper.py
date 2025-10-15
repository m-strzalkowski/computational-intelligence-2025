import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import datetime
import os

years = [str(year) for year in range(2016, 2024)]
starts = ["0101", "0401", "0701", "1001"]
ends = ["0331", "0630", "0930", "1231"]

data = []

for year in years:
    for start, end in zip(starts, ends):
        print(f"Processing {year} from {start} to {end}")
        url = f"https://wykoppl-informacjanieprawdziwa.surge.sh/raport1_{year}{start}-{year}{end}/index.html"
        response = requests.get(url)
        if response.status_code != 200:
            print(response.status_code)
            print(f"Failed to fetch the URL: {url}")
            continue

        soup = BeautifulSoup(response.content, "html.parser")

        posts = soup.find_all("div", style=lambda v: v and "padding-left" in v)
        print("Found", len(posts), "posts")

        for post in posts:
            # Data
            date_tag = post.find("span", title="Data dodania znaleziska")
            date = date_tag.text.strip() if date_tag else None

            # Tytuł i link (pierwszy <a> po "margin-left: 150px")
            main_block = post.find(
                "div", style=lambda v: v and "margin-left: 150px" in v
            )
            title_tag = None
            link = None
            title = None
            if main_block:
                title_tag = main_block.find_all("a", href=True)
                if len(title_tag) > 0:
                    title = title_tag[0].text.strip()
                    link = title_tag[0]["href"]

            # Zawartość
            content_tag = post.find("p", class_="text")
            content = content_tag.text.strip() if content_tag else None

            # Komunikat
            message_tag = post.find(
                "div", style=lambda v: v and "background-color" in v
            )
            message = message_tag.text.strip() if message_tag else None

            # Tagi
            tags_span = post.find("span", title="Tagi")
            tags = (
                tags_span.text.strip().replace("#", "").replace("  ", " ").strip()
                if tags_span
                else None
            )
            if tags:
                tags = ", ".join(tag.strip() for tag in tags.split() if tag)

            # Punkty = Wykopali - Zakopali
            upvote_tag = post.find("a", href=re.compile("/upvotes/up"))
            downvote_tag = post.find("a", href=re.compile("/upvotes/down"))

            def extract_number(tag):
                if tag and tag.text:
                    match = re.search(r"(\d+)", tag.text)
                    return int(match.group(1)) if match else 0
                return 0

            upvotes = extract_number(upvote_tag)
            downvotes = extract_number(downvote_tag)
            points = upvotes - downvotes

            data.append(
                {
                    "date": date,
                    "title": title,
                    "link": link,
                    "message": message,
                    "content": content,
                    "tags": tags,
                    "points": points,
                }
            )
        print(f"Total records so far: {len(data)}")

# Konwersja do DataFrame
df = pd.DataFrame(data)
print(df.head())

date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go one level up and set the data directory
savedir = os.path.join(os.path.dirname(script_dir), "data")

if not os.path.exists(savedir):
    os.makedirs(savedir)

# Save to CSV in the data directory one level above the script
csv_path = os.path.join(savedir, f"mssinfo_wykop_posts_{date_str}.csv")
df.to_csv(csv_path, index=False, encoding="utf-8")
print(f"Zapisano {len(df)} rekordów do {os.path.basename(csv_path)}")
