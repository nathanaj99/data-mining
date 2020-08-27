from pyspark import SparkContext, SparkConf
import json
import time
import sys

sc = SparkContext('local[*]', 'task1')

# Terminal Inputs
input_file = sys.argv[1]
output_file = sys.argv[2]
partition_type = sys.argv[3]
n_partitions = int(sys.argv[4])
n = int(sys.argv[5])

# Reading Data
read_data = sc.textFile(input_file)
review = read_data.map(json.loads).map(lambda x: (x['business_id'], 1))

if partition_type == 'default':
    # --------- DEFAULT PARTITION ---------
    # Partition details
    num_each_partition = review.glom().map(len).collect()
    n_partitions = review.getNumPartitions()
    print(num_each_partition)

    # Actual Result
    t1_start = time.time()
    agg = review.reduceByKey(lambda x, y: x+y).filter(lambda x: x[1] > n).map(lambda x: [x[0], x[1]]).collect()
    t1_end = time.time()

else:
    # --------- CUSTOM PARTITION ---------
    # Use hash function
    def custom_part(id):
        return hash(id)

    review_custom = review.partitionBy(n_partitions, custom_part)

    # Partition Details
    num_each_partition = review_custom.glom().map(len).collect()
    print(num_each_partition)
    print(review_custom.getNumPartitions())

    # Actual result
    t2_start = time.time()
    agg = review_custom.reduceByKey(lambda x, y: x+y).filter(lambda x: x[1] > n).map(lambda x: [x[0], x[1]]).collect()
    t2_end = time.time()


# COMPARISON
# print(t1_end-t1_start) Default partitioner ran for 6.28 s
# print(t2_end-t2_start) Custom partitioner ran for 0.74 s -- significantly faster

# OUTPUT
output = {'n_partitions': n_partitions, 'n_items': num_each_partition, 'result': agg}
output1 = json.dumps(output)

w = open(output_file, 'w')
w.write(output1)
w.close()