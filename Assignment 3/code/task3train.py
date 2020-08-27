from pyspark import SparkContext
import time
import math
import sys
import json
from itertools import combinations
import random

def pearson(x):
    global master_dic
    #global avg
    global list_of_uid
    user_star1 = master_dic[x[0]]
    user_star2 = master_dic[x[1]]
    common_users = list_of_uid[x[0]] & list_of_uid[x[1]]
    ratings_common1 = [user_star1[i] for i in common_users]
    ratings_common2 = [user_star2[i] for i in common_users]
    avg1 = float(sum(ratings_common1))/len(ratings_common1)
    avg2 = float(sum(ratings_common2))/len(ratings_common2)

    num = 0
    den1 = 0
    den2 = 0
    for i in common_users:
        ent1 = user_star1[i] - avg1
        ent2 = user_star2[i] - avg2
        num += ent1 * ent2
        den1 += ent1 * ent1
        den2 += ent2 * ent2

    den = math.sqrt(den1) * math.sqrt(den2)

    if num == 0 or den == 0:
        return None

    p = num/den

    if p >= 0:
        return x[0], x[1], p
    else:
        pass

def make_hash(buckets, coeff, inter, prime):
    def gen_func(x):
        return ((coeff * x + inter) % prime) % buckets #% 308457624821

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
    global num_hash_functions
    list_of_bands = []
    b = 30
    r = math.ceil(float(num_hash_functions)/b)
    remainder = num_hash_functions % b

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

def jaccard_three(x):
    global user_list
    global user_dict2
    global user_business

    u1 = set(user_list[x[0]])
    u2 = set(user_list[x[1]])

    intersect = len(u1 & u2)
    union = len(u1 | u2)
    jac = intersect/union

    if jac >= 0.01:
        if len(user_business[x[0]] & user_business[x[1]]) >= 3:
            return user_dict2[x[0]], user_dict2[x[1]]
        else:
            pass
    else:
        pass

start = time.time()
## ------ CODE STARTS ------
# Read in terminal inputs
train_file = sys.argv[1]
model_file = sys.argv[2]
cf_type = sys.argv[3]

sc = SparkContext('local[*]', 'task3train')
file = sc.textFile(train_file)

initial = file.map(json.loads).map(lambda x: (x['business_id'], x['user_id'], x['stars']))
if cf_type == 'item_based':

    # Dictionary of dictionaries {bid: {uid: rating}}
    master_dic = initial.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().map(lambda x: (x[0], dict(x[1])))

    # Find averages for Pearson correlation
    #avg = master_dic.map(average).collectAsMap()
    #print(avg)

    master_dic = master_dic.collectAsMap()

    #print(master_dic)

    list_of_uid = initial.map(lambda x: (x[0], x[1])).groupByKey().map(lambda x: (x[0], set(x[1]))).collectAsMap()

    business_list = list(master_dic.keys())

    business_pairs = []
    # Generate pairs of businesses with 3+ co-rated users
    for x, y, in combinations(business_list, 2): # 52,556,878 total combinations
        if len(list_of_uid[x] & list_of_uid[y]) >= 3:
            business_pairs.append((x, y))
    #print(len(business_pairs)) # filtered combination total: 1,171,857
    business_pairs = sc.parallelize(business_pairs)

    # find Pearson correlation of all business pairs, filtering out the ones that are negative
    p = business_pairs.map(pearson).filter(lambda x: x is not None).collect()
    #print(len(p))

    with open(model_file, 'w') as w:
        for i in p:
            form = {'b1': i[0], 'b2': i[1], 'sim': i[2]}
            w.write(json.dumps(form) + '\n')
        w.close()

else:
    initial = initial.map(lambda x: (x[1], x[0], x[2]))
    # ----- MINHASH and LSH -----

    # Create user and business dictionaries and associate them with an id "row & column number"
    user_dict = initial.map(lambda x: x[0]).distinct().zipWithIndex()
    user_dict2 = user_dict.map(lambda x: (x[1], x[0]))
    user_dict = user_dict.collectAsMap()
    user_dict2 = user_dict2.collectAsMap()
    #print(len(user_dict2))

    business_dict = initial.map(lambda x: x[1]).distinct().zipWithIndex()
    business_dict2 = business_dict.map(lambda x: (x[1], x[0]))
    business_dict = business_dict.collectAsMap()
    business_dict2 = business_dict2.collectAsMap()
    #print(len(business_dict2))

    # Make User-Business Baskets
    user_business = initial.map(lambda x: (user_dict[x[0]], business_dict[x[1]])).groupByKey().map(lambda x: (x[0], set(x[1])))

    # Dictionary of user-business baskets (based on assigned number, not original id)
    user_list = user_business.collectAsMap()
    #print(len(user_list))

    # Generate n hash functions
    hash_functions = []
    for i in range(30):
        coeff = random.randint(1, 900000000)
        inter = random.randint(1, 900000000)
        prime = 999999937
        hash_functions.append(make_hash(len(business_dict2), coeff, inter, prime))
    num_hash_functions = len(hash_functions)

    # Hash the businesses for every user_id, and take minimum (MINHASH)
    hash_test = user_business.map(one_hash)
    # print(hash_test.collect())

    # LSH: Generate candidate pairs
    candidate_pairs = hash_test.flatMap(partition).groupByKey().map(lambda x: list(x[1])).filter(lambda x: len(x) > 1). \
        flatMap(generate_candidate_pairs).map(lambda x: tuple(sorted(x))).distinct()
    print(candidate_pairs.count())

    # Jaccard similarity >= 0.01 AND at least 3 co-rated businesses
    # with just Jaccard: 17018747
    user_business = user_business.collectAsMap()
    filt = candidate_pairs.map(jaccard_three).filter(lambda x: x is not None)
    print(filt.count())


    # ----- PEARSON CORRELATION -----
    # Read in data from original
    # (uid, (bid, rating)) --> {uid: {bid: rating}}
    master_dic = initial.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().map(lambda x: (x[0], dict(x[1])))
    master_dic = master_dic.collectAsMap()

    # (uid, bid) --> {uid: {bid}}
    list_of_uid = initial.map(lambda x: (x[0], x[1])).groupByKey().map(lambda x: (x[0], set(x[1]))).collectAsMap()

    # find Pearson correlation of all user pairs, filtering out the ones that are negative
    p = filt.map(pearson).filter(lambda x: x is not None).collect()
    #print(len(p)) 542,218

    with open(model_file, 'w') as w:
        for i in p:
            form = {'u1': i[0], 'u2': i[1], 'sim': i[2]}
            w.write(json.dumps(form) + '\n')
        w.close()

end = time.time()
print(end-start)