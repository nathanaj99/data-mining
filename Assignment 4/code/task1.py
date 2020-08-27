import os
from pyspark import SparkContext
from pyspark.sql import SQLContext
from itertools import combinations
from graphframes import *
import sys
import time

# Read terminal inputs
start = time.time()

filter_threshold = int(sys.argv[1])
input_file = sys.argv[2]
output_file = sys.argv[3]

os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 pyspark-shell")

sc = SparkContext('local[*]', 'task1')
sql = SQLContext(sc)

lines = sc.textFile(input_file).map(lambda x: x.split(',')).map(lambda x: (x[0], x[1]))

# Find nodes and edges with at least 7 corated businessed among the user pairs
user_business = lines.groupByKey().map(lambda x: (x[0], set(x[1]))).collectAsMap()
user_list = user_business.keys() # 3375 unique users

user_pairs = [] # 498 pairs
for x, y in combinations(user_list, 2):
    if len(user_business[x] & user_business[y]) >= filter_threshold:
        user_pairs.append((x, y))

nodes = sc.parallelize(user_pairs).flatMap(lambda x: list(x)).distinct().map(lambda x: (x,)) # 222 distinct users

#reference = nodes.collectAsMap()
#print(reference)

nodes = nodes.toDF(['id'])

edges = sc.parallelize(user_pairs)
edges2 = edges.map(lambda x: (x[1], x[0]))
merged = edges.union(edges2).toDF(['src', 'dst'])
#.map(lambda x: (reference[x[0]], reference[x[1]]))

# Create the graphframe
g = GraphFrame(nodes, merged)
communities = g.labelPropagation(maxIter=5).rdd.map(tuple).map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: sorted(list(x[1])))\
    .map(lambda x: (len(x), x)).groupByKey().map(lambda x: (x[0], sorted(list(x[1])))).sortByKey().collectAsMap()

with open(output_file, 'w') as w:
    for i in communities:
        for j in communities[i]:
            if i == 1:
                w.write('\'' + j[0] + '\'' + '\n')
            else:
                for k in j[:-1]:
                    w.write('\'' + k + '\'' + ', ')
                w.write('\'' + j[-1] + '\'' + '\n')
    w.close()

end = time.time()
print(end-start)
