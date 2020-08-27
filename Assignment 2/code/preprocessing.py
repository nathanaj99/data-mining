from pyspark import SparkContext
from pyspark.sql import SQLContext
import json
import csv

sc = SparkContext('local[*]', 'preprocessing')

# Reading data
read_data = sc.textFile('business.json')
business = read_data.map(json.loads).map(lambda x: (x['business_id'], x['state']))
nv = business.filter(lambda x: x[1] == 'NV')

read2 = sc.textFile('review.json')
review = read2.map(json.loads).map(lambda x: (x['business_id'], x['user_id']))

join = review.rightOuterJoin(nv)
join2 = review.join(nv)
print(join.count())
print(join2.count())


sqlContext = SQLContext(sc)
final = join.map(lambda x: (x[1][0], x[0])).toDF(schema=["user_id", "business_id"])
final.coalesce(1).write.csv('output.csv')

with open('output.csv/part-00000-6da1c1ec-1338-4864-918a-bbc1a1dcda08-c000.csv', newline='') as f:
    r = csv.reader(f)
    data = [line for line in r]
with open('output.csv/part-00000-6da1c1ec-1338-4864-918a-bbc1a1dcda08-c000.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['user_id', 'business_id'])
    w.writerows(data)
