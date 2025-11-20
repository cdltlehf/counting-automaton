import math
import re
import statistics as stats

def percentage(num_lst, denom_lst):
    return round(len(num_lst) * 100 / len(denom_lst), 3)

"""
    Perform quantifier analysis of Snort and Polyglot Corpus.
"""
if __name__ == "__main__":
    file_paths = ["../data/filtered/all_regexes.txt", "../data/filtered/snort.txt"]

    lbs, ubs, lbs_wou = [], [], []
    sparse_max, sparse_frac = [], []

    quantifer_count = 0
    for file_path in file_paths:
        file = open(file_path, "r")

        quantifier_capture = re.compile(r"\{[0-9]*,[0-9]*\}")

        for line in file:
            quantifiers = quantifier_capture.findall(line)
            quantifer_count += len(quantifiers)
            for quantifer in quantifiers:
                numbers = quantifer[1:-1]
                bounds = numbers.split(",")

                if bounds[1] != "":
                    l = int(bounds[0])
                    h = int(bounds[1])

                    if h != 0:
                        lbs.append(l)
                        ubs.append(h)

                        if l == 0 or l == 1:
                            sparse_max.append(1)
                            sparse_frac.append((1/h) * 100)

                        else:
                            k = h - l + 1
                            H_ = math.floor((h - 1)/(k + 1)) + math.ceil((h - 1)/(k + 1)) + 1
                            sparse_max.append(H_)
                     
                            sparse_frac.append((H_/h) * 100)
                else:
                    lbs_wou.append(int(bounds[0]))

                    l = int(bounds[0])

                    if l != 0 and l != 1:
                        sparse_max.append(1)
                        sparse_frac.append((1/(int(bounds[0]) - 1)) * 100)
        
        file.close()

    print(f"Number of quantifiers {quantifer_count}")

    lbs_le_64 = list(filter(lambda x: x <= 64, lbs))
    lbs_le_100 = list(filter(lambda x: x <= 100, lbs))
    lbs_le_200 = list(filter(lambda x: x <= 200, lbs))
    lbs_le_1000 = list(filter(lambda x: x <= 1000, lbs))

    print("Lower Bounds Statistics Breakdown:\n")
    print(f"\t Lower Bounds (<= 64): {percentage(lbs_le_64, lbs)}%")
    print(f"\t Lower Bounds (<= 100): {percentage(lbs_le_100, lbs)}%")
    print(f"\t Lower Bounds (<= 200): {percentage(lbs_le_200, lbs)}%")
    print(f"\t Lower Bounds (<= 1000): {percentage(lbs_le_1000, lbs)}%")

    ubs_le_64 = list(filter(lambda x: x <= 64, ubs))
    ubs_le_100 = list(filter(lambda x: x <= 100, ubs))
    ubs_le_200 = list(filter(lambda x: x <= 200, ubs))
    ubs_le_1000 = list(filter(lambda x: x <= 1000, ubs))

    print("\nUpper Bounds Statistics Breakdown:\n")
    print(f"\t Upper Bounds (<= 64): {percentage(ubs_le_64, ubs)}%")
    print(f"\t Upper Bounds (<= 100): {percentage(ubs_le_100, ubs)}%")
    print(f"\t Upper Bounds (<= 200): {percentage(ubs_le_200, ubs)}%")
    print(f"\t Upper Bounds (<= 1000): {percentage(ubs_le_1000, ubs)}%")


    lbs_wou_le_64 = list(filter(lambda x: x <= 64, lbs_wou))
    lbs_wou_le_100 = list(filter(lambda x: x <= 100, lbs_wou))
    lbs_wou_le_200 = list(filter(lambda x: x <= 200, lbs_wou))
    lbs_wou_le_1000 = list(filter(lambda x: x <= 1000, lbs_wou))
    lbs_wou_le_2000 = list(filter(lambda x: x <= 2000, lbs_wou))

    print("\nUnbounded Statistics Breakdown:\n")
    print(f"\t Total: {len(lbs_wou) * 100 /quantifer_count }")
    print(f"\t Lower Bounds (<= 64): {percentage(lbs_wou_le_64, lbs_wou)}%")
    print(f"\t Lower Bounds (<= 100): {percentage(lbs_wou_le_100, lbs_wou)}%")
    print(f"\t Lower Bounds (<= 200): {percentage(lbs_wou_le_200, lbs_wou)}%")
    print(f"\t Lower Bounds (<= 1000): {percentage(lbs_wou_le_1000, lbs_wou)}%")
    print(f"\t Lower Bounds (<= 2000): {percentage(lbs_wou_le_2000, lbs_wou)}%")

    quartiles = stats.quantiles(sparse_frac, n=4)
    print("\nH' / Counting Set Bound:\n")
    print(f"\t Bound Min: {min(sparse_frac)}")
    print(f"\t Qrt. 1: {quartiles[0]}")
    print(f"\t Bound Median: {stats.median(sparse_frac)}")
    print(f"\t Qrt. 3: {quartiles[2]}")
    print(f"\t Bound Max: {max(sparse_frac)}")
    
    sparse_max_le_10 = list(filter(lambda x: x <= 10, sparse_max))
    sparse_max_le_100 = list(filter(lambda x: x <= 100, sparse_max))
    sparse_max_le_200 = list(filter(lambda x: x <= 200, sparse_max))
    sparse_max_le_500 = list(filter(lambda x: x <= 500, sparse_max))
    sparse_max_le_1000 = list(filter(lambda x: x <= 1000, sparse_max))
    
    print("\nH':\n")
    print(f"\t Sparse Maxmimum (<= 10): {percentage(sparse_max_le_10, sparse_max)}%")
    print(f"\t Sparse Maxmimum (<= 100): {percentage(sparse_max_le_100, sparse_max)}%")
    print(f"\t Sparse Maxmimum (<= 200): {percentage(sparse_max_le_200, sparse_max)}%")
    print(f"\t Sparse Maxmimum (<= 500): {percentage(sparse_max_le_500, sparse_max)}%")
    print(f"\t Sparse Maxmimum (<= 1000): {percentage(sparse_max_le_1000, sparse_max)}%")