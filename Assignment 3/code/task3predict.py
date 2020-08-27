from pyspark import SparkContext
from operator import itemgetter
import json
import time
import sys

def predict_item(x):
    global dic1
    global model
    uid = x[0]
    bid = x[1]
    reviewed_business = dic1[x[0]].keys()
    #print(reviewed_business)
    initial = {}
    for i in reviewed_business:
        tup = tuple(sorted((bid, i)))
        if tup in model:
            initial[tup] = model[tup]

    top_n = dict(sorted(initial.items(), key=itemgetter(1), reverse=True)[:5])

    reviewed_n = []
    for i in reviewed_business:
        if tuple(sorted((bid, i))) in top_n:
            reviewed_n.append(i)

    #print(reviewed_n)

    num = 0
    den = 0
    for i in reviewed_n:
        num += (dic1[uid][i]) * (model[tuple(sorted((bid, i)))])
        den += model[tuple(sorted((bid, i)))]

    if den == 0:
        return None

    pred = num/den
    return uid, bid, pred

def predict_user(x):
    global dic1
    global model
    global dic2
    uid = x[0]
    bid = x[1]
    if bid not in dic1:
        return None

    reviewed_users = dic1[bid]
    #print(reviewed_business)
    initial = {}
    for i in reviewed_users:
        tup = tuple(sorted((uid, i)))
        if tup in model:
            initial[tup] = model[tup]
    #print(initial)
    top_n = dict(sorted(initial.items(), key=itemgetter(1), reverse=True)[:5])

    reviewed_n = []
    for i in reviewed_users:
        if tuple(sorted((uid, i))) in top_n:
            reviewed_n.append(i)

    #print(reviewed_n)
    a = []
    for i in dic2[uid]:
        if i != bid:
            a.append(dic2[uid][i])
    if len(a) == 0:
        return None
    ra = float(sum(a))/len(a)
    #a = list(dic2[uid].values())

    #print(ra)

    num = 0
    den = 0
    for i in reviewed_n:
        #print(i)
        u = dic2[i]
        ratings = []
        for j in u:
            if j != bid:
                ratings.append(u[j])
        #print(ratings)
        #if len(ratings) == 0:
            #return None
        ru = float((sum(ratings)))/len(ratings)
        #print(ru)
        num += (u[bid] - ru) * (model[tuple(sorted((uid, i)))])
        den += model[tuple(sorted((uid, i)))]
        #print(num, den)

    if den == 0:
        return None
    pred = ra + num/den

    return uid, bid, pred


## ------- CODE STARTS -------
# Read terminal inputs
train_file = sys.argv[1]
test_file = sys.argv[2]
model_file = sys.argv[3]
output_file = sys.argv[4]
cf_type = sys.argv[5]

sc = SparkContext('local[*]', 'task3predict')
# Read in Test set
test = sc.textFile(test_file).map(json.loads).map(lambda x: (x['user_id'], x['business_id']))  # 58480
print(test.count())
\
if cf_type == 'item_based':
    ## READ IN ALL THE DATA
    # Read in the model from train, convert to dictionary of tuples
    model = sc.textFile(model_file).map(json.loads).map(lambda x: (tuple(sorted((x['b1'], x['b2']))), x['sim'])).distinct().collectAsMap() #692549
    print(len(model))

    # Read in train_set as the master dictionary
    dic1 = sc.textFile(train_file).map(json.loads).map(lambda x: (x['user_id'], (x['business_id'], x['stars'])))\
        .groupByKey().map(lambda x: (x[0], dict(x[1]))).collectAsMap()

    ## ALGORITHM STARTS
    pred = test.map(predict_item).filter(lambda x: x is not None)
    print(pred.count()) # 42712
    print(pred.collect())

else:
    initial = sc.textFile(train_file).map(json.loads).map(lambda x: (x['user_id'], x['business_id'], x['stars']))
    dic1 = initial.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: (x[0], list(x[1]))).collectAsMap()
    #print(dic1)
    dic2 = initial.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().map(lambda x: (x[0], dict(x[1]))).collectAsMap()
    #print(dic2)

    model = sc.textFile(model_file).map(json.loads).map(lambda x: (tuple(sorted((x['u1'], x['u2']))), x['sim'])).distinct().collectAsMap()
    #print(len(model))

    pred = test.map(predict_user).filter(lambda x: x is not None)
    print(pred.count()) #29347
    print(pred.collect())

w = open(output_file, 'a')
for i in pred.collect():
    form = {'user_id': i[0], 'business_id': i[1], 'stars': i[2]}
    json.dump(form, w)
    w.write('\n')
w.close()