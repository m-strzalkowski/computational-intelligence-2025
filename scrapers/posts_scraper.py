from wykop_sdk_reloaded.v3.client import AuthClient, WykopApiClient
import json
import os
import time
import threading
import argparse
import traceback
from tqdm import tqdm
import concurrent.futures


def dump(obj, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False)


# --- Configuration and paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(script_dir), "data")
posts_dir = os.path.join(data_dir, "posts")
os.makedirs(posts_dir, exist_ok=True)


# --- Authentication (app-level) ---
auth = AuthClient()

try:
    app_key = open("klucz_do_api.txt", encoding="utf-8").read().strip()
    app_secret = open("secret_key.txt", encoding="utf-8").read().strip()
except Exception:
    raise RuntimeError(
        "Could not read API key files. Ensure klucz_do_api.txt and secret_key.txt exist."
    )

"""
autoryzując się w ten sposob masz tylko dostep do operacji odczytu.
Reszta wymaga WykopApiClient.authenticate_user()
"""
auth.authenticate_app(app_key, app_secret)


# Thread-local storage so each thread can create a client if needed (safer)
_thread_local = threading.local()


def get_thread_api():
    """Return a WykopApiClient instance stored per-thread to avoid sharing internal state."""
    api = getattr(_thread_local, "api", None)
    if api is None:
        api = WykopApiClient(auth)
        _thread_local.api = api
    return api


def fetch_post(post_id, retries=3, backoff=1.0, pause=0.0):
    """Fetch a single post safely with retries and exponential backoff.

    Returns the "data" dict on success or None on permanent failure.
    """
    attempt = 0
    while attempt <= retries:
        try:
            if pause:
                time.sleep(pause)
            api = get_thread_api()
            resp = api.links_get_link(post_id)
            # expected shape: {"data": {...}}
            return resp.get("data") if isinstance(resp, dict) else resp
        except Exception:
            attempt += 1
            if attempt > retries:
                # give up and return None
                # Print traceback to help debugging but continue processing
                traceback.print_exc()
                print("Moving on...")
                return None
            time.sleep(backoff * (2 ** (attempt - 1)))


def combine_batches_and_dump(out_fname):
    all_posts = {"data": []}
    for filename in os.listdir(posts_dir):
        if filename.startswith("homepage_posts_data_up_to_") and filename.endswith(
            ".json"
        ):
            try:
                with open(
                    os.path.join(posts_dir, filename), "r", encoding="utf-8"
                ) as f:
                    batch = json.load(f)
                    all_posts["data"].extend(batch.get("data", []))
            except Exception:
                # ignore corrupt files but print traceback
                traceback.print_exc()
    dump(all_posts, out_fname)


def main(
    homepage_ids,
    workers=8,
    batch_size=10,
    retries=3,
    backoff=1.0,
    pause=0.0,
    out_file=None,
):
    """Parallelized loader.

    - workers: number of threads
    - batch_size: how many completed items to accumulate before writing an intermediate file
    - retries/backoff: retry policy
    - pause: optional small delay before each request to help rate limiting
    """
    if out_file is None:
        out_file = os.path.join(data_dir, "homepage_posts_data.json")

    total = len(homepage_ids)

    lock = threading.Lock()
    batch_posts = {"data": []}
    completed = 0
    errors = []

    # Choose a reasonable default for workers if not provided
    if workers is None:
        workers = min(32, (os.cpu_count() or 4) * 5)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(fetch_post, pid, retries, backoff, pause): pid
            for pid in homepage_ids
        }
        pbar = tqdm(total=total, desc="Pobieranie danych z postów z id")

        for fut in concurrent.futures.as_completed(futures):
            pid = futures[fut]
            try:
                data = fut.result()
                if data is None:
                    errors.append(pid)
                else:
                    with lock:
                        batch_posts["data"].append(data)

            except Exception:
                errors.append(pid)
                traceback.print_exc()

            completed += 1
            # save intermediate files periodically to avoid data loss / memory growth
            if completed % batch_size == 0:
                idx = completed
                fname = os.path.join(posts_dir, f"homepage_posts_data_up_to_{idx}.json")
                with lock:
                    dump(batch_posts, fname)
                    batch_posts = {"data": []}
                # ensure thread-local batch_posts used after changing
                # (we rebind local variable, safe in this scope)

            pbar.update(1)

        pbar.close()

        # write any remaining
        if batch_posts.get("data"):
            fname = os.path.join(posts_dir, f"homepage_posts_data_up_to_{total}.json")
            dump(batch_posts, fname)

    # combine all and write final file
    combine_batches_and_dump(out_file)

    if errors:
        print(f"Finished with {len(errors)} failed ids. See printed tracebacks above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel posts scraper")
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of parallel worker threads"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="How many items to write per intermediate file",
    )
    parser.add_argument("--retries", type=int, default=3, help="Retries per request")
    parser.add_argument(
        "--backoff", type=float, default=1.0, help="Base backoff seconds"
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="Optional pause (seconds) before each request to help rate limiting",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(data_dir, "homepage_posts_data.json"),
        help="Final output file",
    )

    args = parser.parse_args()

    with open(
        os.path.join(data_dir, "homepage_wykop_ids.json"), "r", encoding="utf-8"
    ) as f:
        homepage_ids = json.load(f).get("ids", [])

    main(
        homepage_ids,
        workers=args.workers,
        batch_size=args.batch_size,
        retries=args.retries,
        backoff=args.backoff,
        pause=args.pause,
        out_file=args.out,
    )
