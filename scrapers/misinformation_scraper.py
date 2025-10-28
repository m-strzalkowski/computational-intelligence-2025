from bs4 import BeautifulSoup
import requests
import re
import datetime
import os
import json

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
            # Date
            date_tag = post.find("span", title="Data dodania znaleziska")
            added_date = date_tag.text.strip() if date_tag else None

            # Author
            author_tag = post.find("a", href=re.compile(r"/ludzie/"))
            author = author_tag.text.strip() if author_tag else None

            # Title and link
            main_block = post.find(
                "div", style=lambda v: v and "margin-left: 150px" in v
            )
            title = link = None
            if main_block:
                a_tags = main_block.find_all("a", href=True)
                if a_tags:
                    title = a_tags[0].text.strip()
                    link = a_tags[0]["href"]

            # Content
            content_tag = post.find("p", class_="text")
            description = content_tag.text.strip() if content_tag else None

            # Alerts (previously message)
            alert_tag = post.find("div", style=lambda v: v and "background-color" in v)
            alerts = alert_tag.text.strip() if alert_tag else None

            # Tags (as list)
            tags_span = post.find("span", title="Tagi")
            if tags_span:
                tags_text = tags_span.text.strip().replace("#", "").strip()
                tags = [tag.strip() for tag in tags_text.split() if tag]
            else:
                tags = []

            # Votes
            upvote_tag = post.find("a", href=re.compile("/upvotes/up"))
            downvote_tag = post.find("a", href=re.compile("/upvotes/down"))

            def extract_number(tag):
                if tag and tag.text:
                    match = re.search(r"(\d+)", tag.text)
                    return int(match.group(1)) if match else 0
                return 0

            votes = extract_number(upvote_tag)
            downvotes = extract_number(downvote_tag)

            # Extract ID from link if possible
            post_id = None
            if link:
                match = re.search(r"/link/(\d+)/", link)
                if match:
                    post_id = int(match.group(1))

            # Append to list
            data.append(
                {
                    "id": post_id,
                    "url": link,
                    "title": title,
                    "description": description,
                    "added_date": added_date,
                    "author": author,
                    "tags": tags,
                    "alerts": alerts,
                    "votes": votes,
                    "downvotes": downvotes,
                }
            )

        print(f"Total records so far: {len(data)}")

json_data = {"data": data}

# Prepare save path
date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
script_dir = os.path.dirname(os.path.abspath(__file__))
savedir = os.path.join(os.path.dirname(script_dir), "data")

if not os.path.exists(savedir):
    os.makedirs(savedir)

json_path = os.path.join(savedir, "mssinfo_wykop_posts.json")

# Save JSON
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(data)} records to {os.path.basename(json_path)}")
