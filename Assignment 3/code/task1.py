from pyspark import SparkContext
import json
import random
from itertools import combinations
import math
import sys
import time

# ------ FUNCTIONS ------
def make_hash(buckets, coeff, inter, prime):
    def gen_func(x):
        return ((coeff * x + inter) % prime) % buckets

    return gen_func

def one_hash(x):
    global hash_functions
    list = []
    for i in hash_functions:
        result = [i(e) for e in x[1]]
        list.append(min(result))
    return x[0], list

def partition(x):
    bid = x[0]
    global hash_functions
    list_of_bands = []
    b = 100
    r = int(math.ceil(len(hash_functions)/b))
    remainder = len(hash_functions) % b
    start = 0
    for i in range(b-remainder):
        s = [j for j in x[1][start:start+r]]
        tup = ((i, hash(tuple(s))), bid)
        start += r
        list_of_bands.append(tup)

    for i in range(b-remainder, b):
        tup = ((i, hash(tuple([j for j in x[1][start:start+r-1]]))), bid)
        start += r-1
        list_of_bands.append(tup)

    return list_of_bands

def generate_candidate_pairs(x):
    return list(combinations(x, 2))

def verify_candidates(x):
    global business_list

    b1 = set(business_list[x[0]])
    b2 = set(business_list[x[1]])

    intersect = len(b1 & b2)
    union = len(b1 | b2)
    jac = intersect/union

    if jac >= 0.05:
        return x[0], x[1], jac
    else:
        pass

start = time.time()
# ------- CODE STARTS --------
# Read Terminal Inputs
input_file = sys.argv[1]
output_file = sys.argv[2]

sc = SparkContext('local[*]', 'task1')

read = sc.textFile(input_file)
lines = read.map(json.loads).map(lambda x: (x['user_id'], x['business_id']))

# Assign IDs for user_id and business_id
# User_id
user_dict = lines.map(lambda x: x[0]).distinct().zipWithIndex()
user_dict = user_dict.collectAsMap()

# Business_id
business_dict = lines.map(lambda x: x[1]).distinct().zipWithIndex()
business_dict2 = business_dict.map(lambda x: (x[1], x[0]))
business_dict = business_dict.collectAsMap()
business_dict2 = business_dict2.collectAsMap()

# User-Business Baskets
business_user = lines.map(lambda x: (business_dict[x[1]], user_dict[x[0]])).groupByKey().map(lambda x: (x[0], sorted(list(x[1]))))
business_list = business_user.collectAsMap()

# Generate 120 random hash functions
hash_functions = []
for i in range(100):
    coeff = random.randint(1, 9000000000)
    inter = random.randint(1, 9000000000)
    prime = 308457624821
    hash_functions.append(make_hash(len(user_dict), coeff, inter, prime))

# MINHASH
# Hash the businesses for every user_id, and take the minimum of all the hashed values for the signature
hash_test = business_user.map(one_hash)

# LSH to find Candidate Pairs
candidate_pairs = hash_test.flatMap(partition)

# Test against ground truth, and only filter those with > 0.05 Jaccard
test = candidate_pairs.groupByKey().map(lambda x: list(x[1])).filter(lambda x: len(x) > 1).\
    flatMap(generate_candidate_pairs).map(lambda x: tuple(sorted(x))).distinct()

print(candidate_pairs.count())

# Verify Candidate pairs
verify = test.map(verify_candidates).filter(lambda x: x is not None)
print(verify.count())

with open(output_file, 'w') as w:
    for i in verify.collect():
        form = {'b1': business_dict2[i[0]], 'b2': business_dict2[i[1]], 'sim': i[2]}
        w.write(json.dumps(form) + '\n')
    w.close()

end = time.time()
print(end-start)