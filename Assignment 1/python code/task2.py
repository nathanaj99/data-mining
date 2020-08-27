from pyspark import SparkContext, SparkConf
import json
import sys

sc = SparkContext('local[*]', 'task1')

# Terminal Inputs
review_file = sys.argv[1]
business_file = sys.argv[2]
output_file = sys.argv[3]
if_spark = sys.argv[4]
n = int(sys.argv[5])

if if_spark == 'spark':
    # ---------- SPARK -----------
    read_b = sc.textFile(business_file)
    business = read_b.map(json.loads).map(lambda x: (x['business_id'], x['categories']))\
        .filter(lambda x: x[1] is not None)\
        .mapValues(lambda x: x.split(', '))
    business_split = business.flatMap(lambda x: [(x[0], i) for i in x[1]])
    print(str(business_split.take(30)))

    read_r = sc.textFile(review_file)
    review = read_r.map(json.loads).map(lambda x: (x['business_id'], x['stars']))
    print(str(review.take(10)))

    # perform join
    join = business_split.join(review)
    print(str(join.take(20)))

    ignore_business = join.map(lambda x: x[1])
    print(str(ignore_business.take(20)))

    # Find average per key (aggregatebyKey)
    agg = ignore_business.aggregateByKey((0, 0), lambda u, v: (u[0] + v, u[1] + 1), lambda a, b: (a[0] + b[0], a[1] + b[1]))\
        .mapValues(lambda x: x[0]/x[1])

    # Reverse order, then sortByKey, and revert back, sorting by alphabetical order (using group by key)
    order = agg.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: (x[0], sorted(list(x[1])))).sortByKey(False)\
        .flatMap(lambda x: [(i, x[0]) for i in x[1]])

    # Format output
    output = {"result": order.take(n)}

else:
    # ---------- VANILLA -----------
    business = []
    with open("../../Documents/SEMESTER 6.5/INF 553/Homework/data/business.json") as f:
        for obj in f:
            business.append(json.loads(obj))

    review = []
    with open("../../Documents/SEMESTER 6.5/INF 553/Homework/data/review.json") as g:
        for obj in g:
            review.append(json.loads(obj))

    class BusinessID:
        def __init__(self, category):
            self.category = category
            self.rating = []

        def addRating(self, rating):
            self.rating.append(rating)

    dic1 = {}
    for i in business:
        if i['categories'] is not None:
            dic1[i['business_id']] = BusinessID(i['categories'].split(', '))

    for i in review:
        if i['business_id'] in dic1:
            dic1[i['business_id']].addRating(i['stars'])

    dic2 = {}
    for key in dic1:
        for i in dic1[key].category:
            if i not in dic2:
                dic2[i] = []
            else:
                pass

    for key in dic1:
        for i in dic1[key].category:
            for j in dic1[key].rating:
                dic2[i].append(j)

    dic3 = {}
    # Loop through each entry in dictionary
    for key in dic2:
        if len(dic2[key]) != 0:
            dic3[key] = sum(dic2[key]) / len(dic2[key])

    final = sorted(dic3.items(), key=lambda x: (-x[1], x[0]))
    ## FORMATTING
    final1 = []
    #for i in final:
        #final1.append([i[0], i[1]])
    for i in range(n):
        final1.append([final[i][0], final[i][1]])
    output = {"result": final1}

output1 = json.dumps(output)
f = open(output_file, 'w')
f.write(output1)
f.close()