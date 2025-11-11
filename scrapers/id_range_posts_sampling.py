import asyncio
import glob
import httpx
import json
import numpy as np
import random
import os
from tqdm.asyncio import tqdm
from bs4 import BeautifulSoup
import itertools
import aiofiles
from concurrent.futures import ThreadPoolExecutor

# --- Paths ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
home_dir = os.path.dirname(script_dir)
data_dir = os.path.join(home_dir, "data")
posts_dir = os.path.join(data_dir, "posts")
os.makedirs(posts_dir, exist_ok=True)

# --- Proxy ---
try:
    with open(os.path.join(script_dir, "proxies.txt"), "r", encoding="utf-8") as pf:
        PROXIES = [line.strip() for line in pf if line.strip()]
except FileNotFoundError:
    PROXIES = []

if not PROXIES:
    raise RuntimeError(
        "No proxies found in proxies.txt — please provide at least one proxy."
    )

# --- Constants ---
BASE_URL = "https://wykop.pl/link/"
CONCURRENCY = 10
TIMEOUT = 15.0
MAX_RETRIES = 5
CPU_WORKERS = 4

executor = ThreadPoolExecutor(max_workers=CPU_WORKERS)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/125.0 Safari/537.36"
}

proxy_cycle = itertools.cycle(PROXIES)


def parse_page(html: str, link_id: int, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    # --- Main post data and alert extraction (Unchanged) ---
    alert_data = None
    alert_section = soup.find("section", class_="warning")
    if alert_section:
        header_el = alert_section.find("h5")
        content_el = alert_section.find("div", class_="content")
        alert_data = {
            "title": header_el.get_text(" ", strip=True) if header_el else None,
            "body": content_el.decode_contents().strip() if content_el else None,
        }
    title_el = soup.select_one("h1.heading a")
    title = title_el.get_text(strip=True) if title_el else None

    # --- Main Post Points (also updated for robustness) ---
    main_points = None
    points_el = soup.select_one("div.dig span")
    if points_el:
        try:
            main_points = int(points_el.get_text(strip=True))
        except (ValueError, TypeError):
            main_points = None

    desc_el = soup.select_one("section.info p")
    description = desc_el.get_text(strip=True) if desc_el else None
    date_el = soup.select_one("time.date")
    added_date = date_el.get("datetime") if date_el else None
    tags = [a.get_text(strip=True) for a in soup.select("li.tag a")]

    # --- GENERALIZED Comments and Replies Parsing Logic ---
    comments_data = []

    comments_container = soup.select_one("div.comments-stream")
    if not comments_container:
        comments_container = soup

    for article in comments_container.select("article[data-v-3dac138d]"):
        if article.find_parent("section[class*='reply']"):
            continue

        # --- Top-level comment parsing ---
        user = (
            el.get_text(strip=True)
            if (el := article.select_one("a.username"))
            else None
        )
        date = el.get("datetime") if (el := article.select_one("time.date")) else None
        content = (
            el.get_text(" ", strip=True)
            if (el := article.select_one("div.wrapper"))
            else None
        )

        # --- CORRECTED Points Parsing for Comments ---
        comment_points = None
        comment_points_el = article.select_one("section.rating-box ul li")
        if comment_points_el:
            try:
                comment_points = int(comment_points_el.get_text(strip=True))
            except (ValueError, TypeError):
                comment_points = None

        # --- Reply parsing ---
        replies = []
        reply_container = article.find_next_sibling("div")

        if reply_container:
            for reply_section in reply_container.select("section[class*='reply']"):
                r_user = (
                    el.get_text(strip=True)
                    if (el := reply_section.select_one("a.username"))
                    else None
                )
                r_date = (
                    el.get("datetime")
                    if (el := reply_section.select_one("time.date"))
                    else None
                )
                r_content = (
                    el.get_text(" ", strip=True)
                    if (el := reply_section.select_one("div.wrapper"))
                    else None
                )

                # --- CORRECTED Points Parsing for Replies ---
                r_points = None
                r_points_el = reply_section.select_one("section.rating-box ul li")
                if r_points_el:
                    try:
                        r_points = int(r_points_el.get_text(strip=True))
                    except (ValueError, TypeError):
                        r_points = None

                replies.append(
                    {
                        "user": r_user,
                        "date": r_date,
                        "content": r_content,
                        "points": r_points,
                    }
                )

        comments_data.append(
            {
                "user": user,
                "date": date,
                "content": content,
                "points": comment_points,
                "replies": replies,
            }
        )

    return {
        "id": link_id,
        "url": url,
        "title": title,
        "points": main_points,
        "description": description,
        "added_date": added_date,
        "tags": tags,
        "alerts": alert_data,
        "comments": comments_data,
    }


# MODIFIED: This function now works with the client pool
async def fetch_and_parse(client_pool, url, link_id, proxy):
    # 1. "Check out" a client from the pool
    client = await client_pool.get()
    try:
        # 2. Configure this specific client with the proxy for this task
        proxy_config = {"http://": proxy, "https://": proxy}
        client.proxies = proxy_config

        for attempt in range(MAX_RETRIES):
            try:
                await asyncio.sleep(random.uniform(0.1, 0.5))

                resp = await client.get(url, timeout=TIMEOUT, follow_redirects=True)

                resp.raise_for_status()

                loop = asyncio.get_running_loop()
                data = await loop.run_in_executor(
                    executor, parse_page, resp.text, link_id, url
                )

                data["status"] = resp.status_code
                data["proxy"] = proxy
                return data

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait = (2**attempt) + random.random()
                    await asyncio.sleep(wait)
                    continue
                return {
                    "id": link_id,
                    "status": e.response.status_code,
                    "url": url,
                    "proxy": proxy,
                    "error": f"HTTP Error: {e.response.status_code}",
                }

            except httpx.RequestError as e:
                wait = (2**attempt) + random.random()
                await asyncio.sleep(wait)
                if attempt == MAX_RETRIES - 1:
                    return {
                        "id": link_id,
                        "status": None,
                        "url": url,
                        "proxy": proxy,
                        "error": str(e),
                    }

        return {
            "id": link_id,
            "status": None,
            "url": url,
            "proxy": proxy,
            "error": f"Failed after {MAX_RETRIES} retries.",
        }

    finally:
        # 3. CRITICAL: Return the client to the pool so another task can use it
        await client_pool.put(client)


def merge_jsonl_files_streamed(min_id, max_id):
    """
    Merges all .jsonl files from the posts_dir into a single, final JSON file
    by streaming the data, which uses very little memory.
    """
    final_output_path = os.path.join(
        data_dir, f"final_combined_results_{min_id}-{max_id}.json"
    )
    jsonl_files = glob.glob(os.path.join(posts_dir, "*.jsonl"))

    if not jsonl_files:
        print("Merge step skipped: No .jsonl files found.")
        return

    print("-" * 50)
    print(f"Starting merge of {len(jsonl_files)} .jsonl files...")

    try:
        with open(final_output_path, "w", encoding="utf-8") as f_out:
            # Write the opening part of the JSON structure
            f_out.write('{\n  "data": [\n')

            is_first_record = True
            for file_path in tqdm(jsonl_files, desc="Merging files"):
                with open(file_path, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        # Skip empty or whitespace-only lines
                        line = line.strip()
                        if not line:
                            continue

                        # Add a comma before every record except the very first one
                        if not is_first_record:
                            f_out.write(",\n")

                        # Write the actual JSON object (line) with indentation
                        f_out.write(f"    {line}")

                        is_first_record = False

            # Write the closing part of the JSON structure
            f_out.write("\n  ]\n}\n")

    except IOError as e:
        print(f"\nError during merge process: {e}")
        return

    print(f"\n✅ Merge complete! Final file saved to: {final_output_path}")


async def main():
    # --- Batching Configuration ---
    batch_size = 1000  # 1000
    number_of_batches = 2  # 20 - adjust if stopped or depending on your needs

    # Load existing IDs to exclude
    def open_json(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    path = "./data/mssinfo_wykop_posts.json"

    data = open_json(path)
    existing_ids = set([p["id"] for p in data["data"]])

    # Create full range and find allowed IDs

    # If for some reason the downloading was surpressed, use MIN_THRESHOLD for the minimum id number
    MIN_THRESHOLD = 7022631
    # MIN_THRESHOLD = min(existing_ids)

    max_id = max(existing_ids)  # extracted from earliest homepage scrape
    min_id = min(existing_ids)  # Wider range for sampling

    start_range = max(min_id, MIN_THRESHOLD)

    full_range = np.arange(start_range, max_id + 1, dtype=int)
    allowed = np.setdiff1d(
        full_range, np.array(list(existing_ids), dtype=int), assume_unique=False
    )

    if allowed.size == 0:
        raise ValueError(
            "No allowed IDs available after excluding existing ids in the given range."
        )

    # Sample uniformly without replacement if possible
    total_samples = batch_size * number_of_batches
    replace = total_samples > allowed.size
    sampled = np.sort(np.random.choice(allowed, size=total_samples, replace=replace))
    batches = sampled.reshape(number_of_batches, batch_size)

    # --- 1. CREATE RESOURCES ONCE, OUTSIDE THE LOOP ---
    client_pool = asyncio.Queue()
    client_list = []
    for _ in range(CONCURRENCY):
        client = httpx.AsyncClient(headers=headers, http2=True)
        await client_pool.put(client)
        client_list.append(client)
    # ---------------------------------------------------

    all_output_paths = []

    try:
        # --- 2. LOOP THROUGH BATCHES ---
        for batch_idx, batch in enumerate(batches):
            ids_to_scrape = batch
            start_id = int(batch.min())
            stop_id = int(batch.max())
            total_ids = len(ids_to_scrape)

            output_path = os.path.join(
                posts_dir, f"results_batch{batch_idx}_{start_id}-{stop_id}.jsonl"
            )
            all_output_paths.append(output_path)

            print("-" * 50)
            print(
                f"Starting batch {batch_idx}: {total_ids} IDs (range {start_id} to {stop_id})."
            )
            print(f"Results will be saved to: {output_path}")

            async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
                tasks = [
                    fetch_and_parse(
                        client_pool,
                        f"{BASE_URL}{int(link_id)}",
                        int(link_id),
                        next(proxy_cycle),
                    )
                    for link_id in ids_to_scrape
                ]

                progress_bar = tqdm(
                    asyncio.as_completed(tasks),
                    total=total_ids,
                    desc=f"Scraping Batch {batch_idx}",
                )
                for future in progress_bar:
                    result = await future
                    if result:
                        await f.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(f"Batch {batch_idx} complete.")

    finally:
        # --- 3. DESTROY RESOURCES ONCE ---
        print("-" * 50)
        print("All batches finished. Shutting down all client connections...")
        for client in client_list:
            await client.aclose()
        # ---------------------------------

    print("\n✅ Scraping complete.")
    print("Results saved to the following files:")
    for path in all_output_paths:
        print(f"- {path}")

    # --- FINAL STEP: MERGE ALL FILES ---
    merge_jsonl_files_streamed(min_id, max_id)

    print("\n✅ All operations complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # The executor shutdown is correctly placed here
        executor.shutdown(wait=True)
