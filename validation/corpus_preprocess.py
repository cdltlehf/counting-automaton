from typing import Optional
import rstr
import json
import re
from multiprocessing import Process, Queue

import cai4py.parser_tools as pt

def create_positive_worker(regex: str, queue: Queue) -> str:
    try:
        candidate = rstr.xeger(regex)
    except:
        print("Xeger error!")
        return

    if re.fullmatch(regex, candidate):
        queue.put(candidate) 
        return

def create_positive(regex: str, timeout: float=8.0) -> Optional[str]:

    queue = Queue()
    p = Process(target=create_positive_worker, args=(regex, queue))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        print("Timed out!")
        return None

    return queue.get() if not queue.empty() else None

def create_negative_worker(regex: str, queue: Queue):

    while True:
        try:
            candidate = rstr.xeger(regex + r"[a-zA-Z0-9]{5,10}")
        except:
            print("Xeger error!")
            return
        
        if not re.fullmatch(regex, candidate):
            queue.put(candidate)
            return

def create_negative(regex: str, timeout: float=8.0) -> Optional[str]:

    queue = Queue()
    p = Process(target=create_negative_worker, args=(regex, queue))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        print("Timed out!")
        return None

    return queue.get() if not queue.empty() else None

if __name__ == "__main__":

    f_corpus = open("../data/filtered/all_regexes.txt", "r")
    width = 800 # number of entries per file

    f_write = open("processed/processed_patterns_0.json", "w")
    entries = []

    pos_skip, neg_skip = 0, 0
    for i, regex in enumerate(f_corpus):

        if i % width == 0 and i != 0:
            json_str = json.dumps(entries, indent=2)
            f_write.write(json_str)
            f_write.close()

            f_write = open(f"processed/processed_patterns_{int(i / width)}.json", "w")
            entries = [] 

        regex = json.loads(regex)

        entry = {"regex": regex}
        positive = create_positive(regex)
        if positive is None:
            pos_skip += 1
            print(f"Positive Skip - cannot generate [{i + 1}]")
        else:
            entry["positive"] = positive

        negative = create_negative(regex)
        if negative is None:
            neg_skip += 1
            print(f"Negative Skip - cannot generate [{i + 1}]")
        else:
            entry["negative"] = negative

        if negative is not None or positive is not None:  
            entries.append(entry)

    json_str = json.dumps(entries, indent=2)
    f_write.write(json_str)
    f_write.close()

    print("Generation statistics:")
    print(f"\t Positive Success Rate: {((i - pos_skip) * 100) / i}%")
    print(f"\t\t Positive Skips: {pos_skip}")
    print(f"\t Negative Success Rate: {((i - neg_skip) * 100) / i}%")
    print(f"\t\t Negative Skips: {neg_skip}")