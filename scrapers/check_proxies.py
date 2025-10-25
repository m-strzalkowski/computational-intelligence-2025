import requests
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

# Load proxies from URL
proxy_list_url = (
    "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt"
)
proxies = requests.get(proxy_list_url).text.splitlines()

test_url = "https://httpbin.dev/"  # website to test
timeout = 5  # seconds


def check_proxy(proxy):
    try:
        response = requests.get(
            test_url,
            proxies={"http": f"http://{proxy}", "https": f"http://{proxy}"},
            timeout=timeout,
        )
        if response.status_code == 200:
            print(f"[WORKING] {proxy}")
            return proxy
    except:
        pass
    return None


# Use ThreadPoolExecutor for parallel checking and stop when we have 10 working proxies
working_proxies = []
executor = ThreadPoolExecutor(max_workers=50)
futures = [executor.submit(check_proxy, p) for p in proxies]

try:
    for fut in as_completed(futures):
        try:
            result = fut.result()
        except Exception:
            result = None
        if result:
            working_proxies.append(result)
            if len(working_proxies) >= 10:
                # cancel remaining futures that haven't started
                for f in futures:
                    if not f.done():
                        f.cancel()
                break
finally:
    # don't block waiting for already-running threads
    executor.shutdown(wait=False)

print(f"\nWorking proxies ({len(working_proxies)}):")
print(working_proxies)
