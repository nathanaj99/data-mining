from pyspark import SparkContext
import json
import binascii
import random
import sys
import math
import time

def bloom_filter(x):
    global hash_functions

    hashed = []
    for i in hash_functions:
        hashed.append(i(x))

    return hashed


def make_hash(buckets, coeff, inter, prime):
    def gen_func(x):
        return ((coeff * x + inter) % prime) % buckets

    return gen_func

# ---- CODE STARTS ----
start = time.time()
# Read Terminal inputs
first_json = sys.argv[1]
second_json = sys.argv[2]
output_file = sys.argv[3]

"""first_json = 'business_first.json'
second_json = 'business_second.json'
output_file = 'output.txt'"""


sc = SparkContext('local[*]', 'task1')

# Change empty string to 'empty' so that it can be converted to integers later
first = sc.textFile(first_json).map(json.loads).map(lambda x: x['city']).distinct().\
    map(lambda x: x if x != '' else 'empty')

n = first.count()
p = 0.0000001

# Determine how many bits and hash functions
num_bits = math.ceil((n * math.log(p)) / math.log(1 / math.pow(2, math.log(2))))
print(num_bits)
num_hash = round((num_bits/n) * math.log(2))
print(num_hash)

# Make hash functions
hash_functions = []
for i in range(num_hash):
    coeff = random.randint(1, 9000000000)
    inter = random.randint(1, 9000000000)
    prime = 308457624821
    hash_functions.append(make_hash(28852, coeff, inter, prime))

# Apply Bloom filter
bloom = first.map(lambda x: int(binascii.hexlify(x.encode('utf8')), 16)).map(bloom_filter).flatMap(lambda x: x).distinct()

# Initialize array of 0's
bitarray = [0] * num_bits

for i in bloom.collect():
    bitarray[i] = 1

ones = {i for i, x in enumerate(bitarray) if x == 1}

tp = 0
fp = 0
tn = 0
fn = 0

list_of_cities = first.collect()
results = []
with open(second_json) as f:
    for line in f:
        test = json.loads(line)['city']
        if test == '':
            test = 'empty'
        # Convert to ascii
        test1 = int(binascii.hexlify(test.encode('utf8')), 16)
        # Pass through all hash_functions
        hashed = set(bloom_filter(test1))

        if hashed.issubset(ones):
            pred = 1
            results.append(1)
        else:
            pred = 0
            results.append(0)

        if test in list_of_cities:
            actual = 1
        else:
            actual = 0

        if pred == 1 and actual == 1:
            tp += 1
        if pred == 1 and actual == 0:
            fp += 1
        if pred == 0 and actual == 0:
            tn += 1
        if pred == 0 and actual == 1:
            fn += 1

with open(output_file, 'w') as w:
    for i in results:
        w.write(str(i) + ' ')

fpr = float(fp)/float((fp+tn))
fnr = float(fn)/float((fn+tp))
tnr = float(tn)/float((tn+fp))
tpr = float(tp)/float((fn+tp))
print(fpr, fnr, tnr, tpr)
end = time.time()

print(end-start)