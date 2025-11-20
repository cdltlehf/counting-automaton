import argparse
import json

from multiprocessing import Process, Queue
from pathlib import Path
import os

import cai4py.counting_automaton.position_counting_automaton as pca
import cai4py.counting_automaton.super_config.super_config as sc
from cai4py.custom_counters.counter_type import CounterType
import cai4py.parser_tools as pt

current_dir = Path(__file__).resolve().parent
processed_dir = current_dir / "processed"

def positive_match(element, matcher, queue):
    
    try:
        positive = element["positive"]
    except:
        queue.put(-10)
        return queue    # didnt find

    try:
        val, _ = matcher.match(positive)
        if val == False:
            queue.put(1)
            return queue   # miss
    except:
        queue.put(-1)   
        return queue    # error

    queue.put(0) 
    return queue    # passed

def negative_match(element, matcher, queue):
    try:
        negative = element["negative"]
    except:
        queue.put(-10)
        return queue    # didnt find

    try:
        val, _ = matcher.match(negative)
        if val == True:
            queue.put(1)
            return queue   # miss
    except:
        queue.put(-1)   
        return queue    # error
    
    queue.put(0) 
    return queue    # passed

# Used for the testing and verification of counter method over the Polyglot corpus

# Parsing of arguments
parser = argparse.ArgumentParser(description="Full demo to match processed regexes.")
parser.add_argument("--count", type=int, default=-1, help="The number of regexes to try and match.")
parser.add_argument("--method", type=int, default=1, help="Select counter methods: 1 - Bit Vector; 2 - Naive Counter; 3 - Counting-set; 4 - Sparse Counting-set; 5 - Counter Expansion")

args = parser.parse_args()

if __name__ == "__main__":

    files = os.listdir(processed_dir)
    files.reverse()

    method = args.method

    # Basic statistics
    count = 0
    pos_miss = 0
    neg_miss = 0
    pos_nc = 0
    neg_nc = 0

    # Method selection
    match method:
        case 1:
            counter = CounterType.BIT_VECTOR
            normaliser = pt.flatten_inner_quantifiers
            method_name = "Bit Vector"
        case 2:
            counter = CounterType.NAIVE_COUNTER
            normaliser = pt.flatten_inner_quantifiers
            method_name = "Lazy Counting Set"
        case 3:
            counter = CounterType.COUNTING_SET
            normaliser = pt.flatten_inner_quantifiers
            method_name = "Counting Set"
        case 4:
            counter = CounterType.SPARSE_COUNTING_SET
            normaliser = pt.flatten_inner_quantifiers
            method_name = "Sparse Counting Set"
        case 5:
            counter = CounterType.BIT_VECTOR
            normaliser = pt.flatten_quantifiers
            method_name = "Counter Expansion"

    print(f"Method: {method_name}")

    i = -1
    for file_name in files:

        if i >= args.count and args.count != -1:
            break

        print(f"\tFile: {file_name}\n")

        file = open(f"{processed_dir}/{file_name}", "r")
        data = json.load(file)

        for element in data:
            i += 1

            if i >= args.count and args.count != -1:
                break

            regex = element["regex"]
            try:
                parsed_regex = pt.parse(regex)
                normalised_regex = normaliser(parsed_regex)
                pattern = pt.to_string(normalised_regex)
            except:
                assert f"Problem parsing/normalising regex: {regex}"

            try:
                automaton = pca.PositionCountingAutomaton.create(pattern)
                matcher = sc.SuperConfig(automaton, counter)
            except:
                assert f"Problem creating automaton for {regex}"

            # Attempt Positive Match
            queue = Queue()
            p = Process(target=positive_match, args=(element, matcher, queue))
            p.start()
            p.join(timeout=10.0)

            if p.is_alive():
                p.terminate()
                p.join()
                print(f"Timed out on Positive Match: {json.dumps(regex)}")
                pos_nc += 1
            else:
                result = queue.get()
                
                if result == -10:
                    pos_nc += 1 # Does not contain positive case

                elif result == 1:
                    print(f"Miss in Positive Match: {json.dumps(regex)}")
                    pos_miss += 1

                elif result == -1:
                    print(f"Error in Positive Match: {json.dumps(regex)}")
                    pos_miss += 1

            # Attempt Negative Match
            queue = Queue()
            p = Process(target=negative_match, args=(element, matcher, queue))
            p.start()
            p.join(timeout=10.0)

            if p.is_alive():
                p.terminate()
                p.join()
                print(f"Timed out on Negative Match: {json.dumps(regex)}")
                neg_nc += 1
            else:
                result = queue.get()
                if result == -10: # Does not contain negative case
                    neg_nc += 1

                elif result == 1:
                    print(f"Miss in Negative Match: {json.dumps(regex)}")
                    neg_miss += 1 

                elif result == -1:
                    print(f"Error in Negative Match: {json.dumps(regex)}")
                    neg_miss += 1

        file.close()
        print("")

    print(f"\nMethod: {method_name}\nTotal Test Cases: {i}")
    print(f"\t Total Positive Cases: {i - pos_nc}")
    print(f"\t Total Negative Cases: {i - neg_nc}")

    print(f"\nTotal Misses: {pos_miss + neg_miss}\n")
       
    print(f"\t Positive Miss Percentage: {(pos_miss*100) / (i - pos_nc)}%")
    print(f"\t\t Positive Misses: {pos_miss}")
       
    print(f"\t Negative Miss Percentage: {(neg_miss*100) / (i - neg_nc)}%")
    print(f"\t\t Negative Misses: {neg_miss}")