from pyspark import SparkContext
import pickle
import json
import sys

# --------- FUNCTIONS ---------
def cosine_similarity(x):
    global user_profiles
    global business_profiles
    uid = x[0]
    bid = x[1]

    if uid not in user_profiles or bid not in business_profiles:
        return None

    cosine = len(user_profiles[uid] & business_profiles[bid])/200

    if cosine >= 0.01:
        return uid, bid, cosine
    else:
        return None

# ------ CODE STARTS ------

# Read Terminal Inputs
test_file = sys.argv[1]
model_file = sys.argv[2]
output_file = sys.argv[3]

# READ IN DATA
# Model (User + Business Profiles)
model = open(model_file, 'rb')

user_profiles = pickle.load(model)
business_profiles = pickle.load(model)

# Test data
sc = SparkContext('local[*]', 'task2predict')
lines = sc.textFile(test_file).map(json.loads).map(lambda x: (x['user_id'], x['business_id']))

# CALCULATIONS
# Cosine similarity calculation
cos = lines.map(cosine_similarity).filter(lambda x: x is not None)

# WRITE OUT
with open(output_file, 'w') as w:
    for i in cos.collect():
        form = {'user_id': i[0], 'business_id': i[1], 'sim': i[2]}
        w.write(json.dumps(form) + '\n')
    w.close()