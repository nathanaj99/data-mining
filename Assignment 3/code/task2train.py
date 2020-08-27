from pyspark import SparkContext
import json
import pickle
import math
from collections import Counter
from operator import itemgetter
from heapq import nlargest
import sys

def preprocessing(x):
    lower = x[1].lower()
    rem = '"*_^{}`+=/~%-$&!@#([,.!?:;])0123456789\\'
    for i in rem:
        lower = lower.replace(i, ' ')
    """backslash = ['\n', '\"']
    for i in backslash:
        lower = lower.replace(i, ' ')"""

    return x[0], lower


def term_freq(x):
    bid = x[0]
    words = x[1]
    count = Counter(words)

    """for i in words:
        if i not in count:
            count[i] = 1
        else:
            count[i] += 1"""

    maximum = max(count, key=count.get)
    max_value = count[maximum]

    for i in count:
        count[i] = count[i] / max_value

    return bid, count


def tfidf(x):
    global idf
    tf = x[1]
    for i in tf:
        if i not in idf:
            print(i)
        else:
            tf[i] = tf[i] * idf[i]

    largest_200 = dict(sorted(tf.items(), key=itemgetter(1), reverse=True)[:200])
    return x[0], largest_200


## --------- CODE STARTS ----------
# Read Terminal Inputs
train_file = sys.argv[1]
model_file = sys.argv[2]
stop = sys.argv[3]


sc = SparkContext('local[*]', 'task1')

read = sc.textFile(train_file)
lines = read.map(json.loads).map(lambda x: (x['user_id'], x['business_id'], x['text']))

stopwords = sc.textFile(stop).flatMap(lambda x: x.split('\t')).collect()
stopwords.append('')

# PRE-PROCESSING
# Removing punctuation, numbers, and stopwords
# Removing extremely rare words (less than 0.0001% of total words) -- STILL NEED TO DO THIS
# after preprocessing map function, filter stop words
bid = lines.map(lambda x: (x[1], x[2]))
pre = bid.map(preprocessing).map(lambda x: (x[0], x[1].split())).flatMap(lambda x: [(x[0], i) for i in x[1]]). \
    filter(lambda x: x[1] not in stopwords).groupByKey().map(lambda x: (x[0], list(x[1])))

count_docs = pre.count()  # 10253 unique businesses

tf = pre.map(term_freq)

idf = pre.map(lambda x: set(x[1])).flatMap(lambda x: [(i, 1) for i in x]).reduceByKey(lambda x, y: x + y). \
    map(lambda x: (x[0], math.log(count_docs / x[1], 2))).collectAsMap()

tfidf = tf.map(tfidf)

# MAKE USER PROFILES
uid = lines.map(lambda x: (x[1], x[0]))
user_profiles = uid.leftOuterJoin(tfidf).map(lambda x: x[1]).groupByKey().map(
    lambda x: (x[0], {k: v for i in list(x[1]) for k, v in i.items()})) \
    .map(lambda x: (x[0], set(nlargest(200, x[1], key=x[1].get)))).collectAsMap()

# MAKE BUSINESS PROFILES
business_profiles = tfidf.map(lambda x: (x[0], set(x[1].keys()))).collectAsMap()

w = open(model_file, 'wb')
pickle.dump(user_profiles, w)
pickle.dump(business_profiles, w)
w.close()