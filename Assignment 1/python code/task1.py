 from pyspark import SparkContext, SparkConf
import json
import sys

#conf = (SparkConf().setMaster("local").setAppName("task1").set("spark.executor.memory", "1g"))
sc = SparkContext('local[*]', 'task1')
input_file = sys.argv[1]
output_file = sys.argv[2]
stopwords = sys.argv[3]
y = sys.argv[4]
m = sys.argv[5]
n = sys.argv[6]

read_data = sc.textFile(input_file)
test = read_data.map(json.loads).map(lambda x: (x['user_id'], x['text'], x['date']))

#   ------- Part A ---------
total = test.count()

#   ------- Part B ---------
# Pick number of distinct

date = test.map(lambda x: (x[2]))  # only take the date
date = date.filter(lambda x: x[0:4] == str(y))
yr_count = date.count()

#   ------- Part C ---------
users = test.map(lambda x: (x[0])).distinct()
dist_users_count = users.count()

#   ------- Part D ---------
# sum all the user reviews
user = test.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a+b)
# sort and filter first m (NEED TO CHANGE: m instead of 10)
user_sort = user.map(lambda x: (x[1], x[0])).sortByKey(False)
user_display = user_sort.map(lambda x: [x[1], x[0]]).take(int(m))

#   ------- Part E ---------
def remove_punc_lower(x):
    lower = x.lower()
    punc = '([,.!?:;])'
    for i in punc:
        lower = lower.replace(i, '')
    return lower

stop_words = sc.textFile(stopwords).flatMap(lambda x: x.split('\t')).collect()


text = test.map(lambda x: (x[1])).map(remove_punc_lower).flatMap(lambda x: x.split(' '))
word_count = text.map(lambda x: (x, 1)).reduceByKey(lambda a, b: a+b)
word_sort = word_count.map(lambda x: (x[1], x[0])).sortByKey(False)
word_display = word_sort.map(lambda x: (x[1], x[0])).filter(lambda x: x[0] not in stop_words).filter(lambda x: x[0] != '').take(int(n))

## OUTPUT IN A JSON
output = {"A": total, "B": yr_count, "C": dist_users_count, "D": [x for x in user_display], "E": [x[0] for x in word_display]}
output1 = json.dumps(output)

w = open(output_file, 'w')
w.write(output1)
w.close()