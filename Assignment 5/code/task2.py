from pyspark import SparkContext
from pyspark.streaming import StreamingContext
import json
import random
import statistics
import time
import sys

def make_hash(buckets, coeff, inter, prime):
    def gen_func(x):
        return ((coeff * x + inter) % prime) % buckets

    return gen_func

def count_trailing(x):
    count = 0
    while (x & 1) == 0:
        x = x >> 1
        count += 1
    return count

def flajolet_martin(x):
    global hash_functions
    hashed = []
    counter = 1
    for i in hash_functions:
        buffer = i(abs(hash(x)))
        trail = count_trailing(buffer)
        hashed.append((counter, trail))
        counter += 1
    return hashed

def write_out(rdd):
    w = open(output_file, 'a')
    w.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())) + ',')
    for entry in rdd.collect():
        w.write(str(entry) + ',')
        w.close()

def write_out_est(rdd):
    w = open(output_file, 'a')
    for entry in rdd.collect():
        w.write(str(entry) + '\n')
        w.close()

## CODE STARTS
# Read in input
port = sys.argv[3]
output_file = sys.argv[2]

w = open(output_file, 'w')
w.write('Time,Ground Truth,Estimation\n')
w.close()

hash_functions = []
for i in range(1000):
    coeff = random.randint(1, 9000000000)
    inter = random.randint(1, 9000000000)
    buckets = 1000000000
    prime = 308457624821
    hash_functions.append(make_hash(buckets, coeff, inter, prime))

sc = SparkContext('local[*]', 'task2')
ssc = StreamingContext(sc, 5) # Batch duration: 5 seconds

read = ssc.socketTextStream('localhost', 9999)
test = read.map(json.loads).map(lambda x: x['city']).window(30, 10)

#time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

ct = test.transform(lambda rdd: rdd.distinct()).count().foreachRDD(write_out)

test = test.transform(lambda rdd: rdd.distinct()).flatMap(flajolet_martin).groupByKey().map(lambda x: (x[0] % 500, 2**max(x[1]))).\
    groupByKey().map(lambda x: (x[0], float(sum(list(x[1]))/len(list(x[1]))))).map(lambda x: (1, x[1])).groupByKey().\
    map(lambda x: sorted(list(x[1]))).map(lambda x: statistics.median(x[:int(len(x)/2)])).foreachRDD(write_out_est)

ssc.start()
ssc.awaitTermination()

