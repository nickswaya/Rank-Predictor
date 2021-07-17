import requests
from concurrent.futures import ThreadPoolExecutor
import timeit




def main():
    with ThreadPoolExecutor(max_workers=2) as executor:
            executor.shutdown(wait=True)

