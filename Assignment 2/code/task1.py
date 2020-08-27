from pyspark import SparkContext
import sys
import math
import time

## ---------- FUNCTIONS -----------
# --- Apriori (Phase 1) ---
def frequent_items(candidates, baskets, s):
    freq_itemsets = []
    for i in candidates:
        counter = 0
        for b in baskets:
            if i.issubset(b):
                counter += 1
        if counter >= s:
            freq_itemsets.append(i)
    return freq_itemsets


def apriori(baskets, candidates1, s):
    freq = []
    # FREQ ITEMSET SIZE 2
    k = 2
    while len(candidates1) > 0:
        buffer = frequent_items(candidates1, baskets, s)
        #print(buffer)
        freq.append(buffer)

        next_candidates = []
        for i, u in enumerate(buffer):
            for v in buffer[i + 1:]:
                if list(u)[:k - 2] == list(v)[:k - 2]:
                    next_candidates.append(u | v)
        candidates1 = next_candidates
        k += 1
    return freq


# --- Count function for Phase 2 ---
def count(part, hi):
    counts_all = []
    for i in hi:
        counter = 0
        make_set = set(p for p in i)
        for j in part:
            if make_set.issubset(j):
                counter += 1
        counts_all.append((i, counter))

    return counts_all


# ------------ CODE STARTS ------------
## --- READ TERMINAL INPUTS ---
case_number = int(sys.argv[1])
support = float(sys.argv[2])
input_file = sys.argv[3]
output_file = sys.argv[4]


# ---- INITIAL READ ----
start = time.time()
sc = SparkContext('local[*]', 'task1')
lines = sc.textFile(input_file).map(lambda x: x.split(','))

# filter out column names
header = lines.first()
lines = lines.filter(lambda x: x != header)

if case_number == 1:
    # create market-basket model (user_id => business_id)
    lines = lines.groupByKey().map(lambda x: set(x[1]))
elif case_number == 2:
    lines = lines.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: set(x[1]))

numParts = lines.getNumPartitions()
supportPart = math.ceil(support/float(numParts))

candidate_items1 = lines.flatMap(lambda x: x).distinct().map(lambda x: {x}).collect()
baskets = lines.count()
# -------- MAP REDUCE PHASE 1 ---------
mr1 = lines.mapPartitions(lambda x: apriori(list(x), candidate_items1, supportPart))\
    .flatMap(lambda x: x).map(lambda x: (tuple(sorted(x)), 1)).groupByKey().map(lambda x: x[0]).collect()

# -------- MAP REDUCE PHASE 2 ---------
mr2 = lines.mapPartitions(lambda x: count(list(x), mr1)).reduceByKey(lambda x, y: x + y).\
    filter(lambda x: x[1] >= support).map(lambda x: x[0]).collect()


## OUTPUT
def format(result, w):
    dictionary_1 = {i: [] for i in range(1, max(map(len, result)) + 1)}
    for i in result:
        dictionary_1[len(i)].append(i)

    dic_sort = {i: [] for i in range(1, max(map(len, result)) + 1)}
    # SORT LEXICOGRAPHICALLY
    for i in dictionary_1:
        for j in dictionary_1[i]:
            set1 = sorted(set(p for p in j))
            dic_sort[i].append(set1)

    dic_sort1 = {i: [] for i in range(1, max(map(len, result)) + 1)}
    for i in dic_sort:
        dic_sort1[i] = sorted(dic_sort[i], key=lambda x: tuple(x[j] for j in range(i)))

    for i in dic_sort1:
        s = ""
        for j in dic_sort1[i][:-1]:
            if i == 1:
                s = s + "('" + j[0] + "')" + ','
            else:
                s = s + str(tuple(j)) + ','
        if i == 1:
            s = s + "('" + dic_sort1[i][-1][0] + "')\n"
        else:
            s = s + str(tuple((dic_sort1[i][-1]))) + "\n"

        w.write(s)


w = open(output_file, 'a')
w.write("Candidates:\n")
format(mr1, w)
w.write("Frequent Itemsets:\n")
format(mr2, w)

w.close()

end = time.time()
print("'Duration: " + str(end-start) + "'")