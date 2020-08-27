from pyspark import SparkContext
import random
import math
import time
import sys
import os
import glob
import json

def compute_distance(p1, p2):
    sum_sq = 0
    for i in range(len(p1)):
        sum_sq += (float(p1[i])-float(p2[i]))**2

    return math.sqrt(sum_sq)

def find_closest_distances(list_of_points, centroids):
    global master_points

    closest_distances = {}
    for point in list_of_points:
        min = sys.maxsize
        for centroid in centroids:
            distance = compute_distance(centroid, master_points[point])
            if distance < min:
                min = distance
        closest_distances[point] = min

    return closest_distances

def find_closest_clusters(list_of_points, centroids):
    global master_points

    closest_cluster = {}
    for point in list_of_points:
        cluster = 0
        min = sys.maxsize
        for centroid in centroids:
            distance = compute_distance(centroid, master_points[point])
            if distance < min:
                min = distance
                cluster = centroid
        if cluster not in closest_cluster:
            closest_cluster[cluster] = [point]
        else:
            closest_cluster[cluster].append(point)

    return closest_cluster

def new_centroids(clusters):
    # Get the existing cluster configuration and get its average
    global master_points
    output = []
    for cluster in clusters:
        coordinates = [master_points[i] for i in clusters[cluster]]
        #print(coordinates)

        new_coordinates = []
        for i in range(len(coordinates[0])):
            ith_dim = [float(tup[i]) for tup in coordinates]
            avg = float(sum(ith_dim))/float(len(ith_dim))
            new_coordinates.append(avg)
        output.append(tuple(new_coordinates))

    return output

def k_means(k, iter, list_of_points):
    # ------ K-MEANS++ -------
    """ USEFUL VARIABLES
    - Centroids = list of COORDINATES, not nodes
    - list_of_points = list of points that are used in the k-means algorithm (not coordinates)

    OUTPUT: Clusters: {centroid coordinates: [list of points (not coordinates) associated]}
    """
    global master_points

    if len(list_of_points) == 0: # Checking condition only if there are no outliers
        return {}

    centroids = []
    ## PROBLEM: Might need to take a sample of the data since it runs so goddamn slow
    ## PROBLEM: Need a way to find out the ideal k
    # Initialize first centroid
    centroids.append(master_points[random.choice(list_of_points)])
    print(centroids)

    # Find the other 19 initial centroids
    for i in range(1, k):
        # finding closest distances to centroid
        closest_distances = find_closest_distances(list_of_points, centroids)
        # print(closest_distances)

        # sample with probability proportional to squared distance
        dist_squared = [i ** 2 for i in list(closest_distances.values())]
        choice = random.choices(list_of_points, weights=dist_squared, k=1)
        #print(choice[0])
        centroids.append(master_points[choice[0]])

    # initial clusters: Find closest centroid
    clusters = find_closest_clusters(list_of_points, centroids)
    #print({k: len(v) for (k, v) in clusters.items()}.values())

    # Now, do k-means iteration
    for i in range(iter):
        centroids = new_centroids(clusters)
        clusters = find_closest_clusters(list_of_points, centroids)
        #print({k: len(v) for (k, v) in clusters.items()}.values())

    return clusters

def summarize_ds(clusters):
    # INPUT: clusters {centroid coordinates, [list of points]}
    # OUTPUT: [(n, sum, sumsq, centroid, sd)]
    output = []
    for cluster in clusters:
        n = len(clusters[cluster])
        list_of_coordinates = [master_points[i] for i in clusters[cluster]]
        sum = []
        sumsq = []
        for i in range(len(list_of_coordinates[0])):
            sum_i = 0
            sumsq_i = 0
            for j in list_of_coordinates:
                sum_i += j[i]
                sumsq_i += j[i] ** 2
            sum.append(sum_i)
            sumsq.append(sumsq_i)

        # CALCULATE centroid and sd
        avg = [float(x)/float(n) for x in sum]
        sumsq_n = [float(x)/float(n) for x in sumsq]
        sd = [math.sqrt(sumsq_n[j] - avg[j]**2) for j in range(len(avg))]

        output.append((n, sum, sumsq, avg, sd))

    return output

def summarize_cs(points):
    global master_points
    n = len(points)
    list_of_coordinates = [master_points[i] for i in points]
    sum = []
    sumsq = []
    for i in range(len(list_of_coordinates[0])):
        sum_i = 0
        sumsq_i = 0
        for j in list_of_coordinates:
            sum_i += float(j[i])
            sumsq_i += float(j[i]) ** 2
        sum.append(sum_i)
        sumsq.append(sumsq_i)

    # CALCULATE centroid and sd
    avg = [float(x) / float(n) for x in sum]
    sumsq_n = [float(x) / float(n) for x in sumsq]
    sd = [math.sqrt(sumsq_n[j] - avg[j] ** 2) for j in range(len(avg))]

    return (n, sum, sumsq, avg, sd)

def update_ds(ds_one, point):
    """
    :param ds_one: (n, sum, sumsq, avg, sd)
    :param point: (x1, x2, ..., xn)
    :return: (n, sum, sumsq, avg, sd)
    """
    n = ds_one[0]
    sum = ds_one[1]
    sumsq = ds_one[2]

    dim = len(point)
    n += 1
    for i in range(dim):
        sum[i] += float(point[i])
        sumsq[i] += (float(point[i]) ** 2)

    # Recalculate centroid and sd
    avg = [float(x) / float(n) for x in sum]
    sumsq_n = [float(x) / float(n) for x in sumsq]
    sd = [math.sqrt(sumsq_n[j] - avg[j] ** 2) for j in range(len(avg))]

    return (n, sum, sumsq, avg, sd)

def mahalanobis(x, c):
    """
    :param x: Point
    :param c: Cluster summary (can be used for ds or cs)
    :return: distance
    """
    centroid = c[0]
    sd = c[1]
    y = 0
    for i in range(len(x)):
        if sd[i] == 0: # Will there be sd of 0 in practice? This raises a division error
            sd[i] = 0.000001
        buff = ((x[i]-centroid[i])/sd[i]) ** 2
        y += buff

    return math.sqrt(y)

def combine_cs(cs1, cs2):
    global cs
    global dim
    # Combine cs
    n = cs1[0] + cs2[0]
    sum = []
    sumsq = []
    for i in range(dim):
        sum.append(float(cs1[1][i]) + float(cs2[1][i]))
        sumsq.append(float(cs1[2][i]) + float(cs2[2][i]))

    # Recalculate centroid and sd
    avg = [float(x) / float(n) for x in sum]
    sumsq_n = [float(x) / float(n) for x in sumsq]
    sd = [math.sqrt(sumsq_n[j] - avg[j] ** 2) for j in range(len(avg))]

    return (n, sum, sumsq, avg, sd)

def merge_to_ds(cs, ds):
    global dim
    # Combine cs
    n = cs[0] + ds[0]
    sum = []
    sumsq = []
    for i in range(dim):
        sum.append(float(cs[1][i]) + float(ds[1][i]))
        sumsq.append(float(cs[2][i]) + float(ds[2][i]))

    # Recalculate centroid and sd
    avg = [float(x) / float(n) for x in sum]
    sumsq_n = [float(x) / float(n) for x in sumsq]
    sd = [math.sqrt(sumsq_n[j] - avg[j] ** 2) for j in range(len(avg))]

    return (n, sum, sumsq, avg, sd)


def output_intermediate(round, ds, cs, rs, w):
    round_id = round
    nof_cluster_discard = len(ds)
    nof_point_discard = sum([x[0] for x in ds])
    nof_cluster_compression = len(cs)
    nof_point_compression = sum([x[0] for x in cs])
    nof_point_retained = len(rs)
    #w = open('output.csv', 'a')
    w.write(str(round_id) + ',' + str(nof_cluster_discard) + ',' + str(nof_point_discard) + ',' +
            str(nof_cluster_compression) + ',' + str(nof_point_compression) + ',' + str(nof_point_retained) + '\n')


start = time.time()
## ------ CODE STARTS -----
# Read Terminal Inputs
"""input_directory = sys.argv[1]
k = int(sys.argv[2])
outfile1 = sys.argv[3]
outfile2 = sys.argv[4]"""

input_directory = 'data/test2'
k = 10
outfile1 = 'output.csv'
outfile2 = 'output2.json'

#os.environ['PYSPARK_PYTHON'] = 'usr/bin/python3.6'
#os.environ['PYSPARK_DRIVER_PYTHON'] = 'usr/bin/python3.6'

owd = os.getcwd()
# Make a list of files to iterate through
os.chdir(input_directory)
list_of_files = []
for file in glob.glob("*.txt"):
    list_of_files.append(file)
list_of_files = sorted(list_of_files)

os.chdir(owd)

## INITIALIZE FINAL OUTPUT
final_output = {} # {point: cluster number}
cs_store = [] # Stores all points currently in the cs. Need to recheck again with final answer to output in json

# OPEN OUTPUT1 FILE
w = open(outfile1, 'w')
w.write('round_id,nof_cluster_discard,nof_point_discard,nof_cluster_compression,nof_point_compression,nof_point_retained\n')

####### ROUND 1 STARTS #########
round = 1
print(round)
sc = SparkContext('local[*]', 'bfr')
read_file = input_directory + '/' + list_of_files[0]

master_points = sc.textFile(read_file).map(lambda x: x.split(',')).map(lambda x: (x[0], tuple([float(i) for i in x[1:]]))).collectAsMap()

dim = len(list(master_points.values())[0])

# STEP 1: First k-means to identify outliers
list_of_points = list(master_points.keys())
clusters = k_means(2*k, 1, list_of_points)

# STEP 2: Check if any of the clusters are outliers (if a cluster only has 1 or 2 points?).
outliers = []
core = []

for i in clusters:
    if len(clusters[i]) == 1: # might need to change the outlier threshold
        for j in clusters[i]:
            outliers.append(j)
    else:
        for j in clusters[i]:
            core.append(j)

# A) For core points, find DS by running clustering again with k. Summarize, then discard
raw_ds = k_means(k, 6, core)
# THIS RAW DATA IS IMPORTANT: Must save this somehow
counter = 0
for i in raw_ds:
    for j in raw_ds[i]:
        final_output[int(j)] = counter
    counter += 1

ds = summarize_ds(raw_ds)
#print([x[0] for x in ds])


# B) For outliers, find CS and RS by running clustering again (this is a little confusing)
out = k_means(2*k, 2, outliers)
cs = [] # list of (summarized) miniclusters that have not yet been merged to a large cluster
rs = [] # list of single points not in any cluster--not coordinates
counter = 0
for i in out:
    if len(out[i]) == 1:
        for j in out[i]:
            rs.append(j)
    else:
        # Summarize each cluster
        for j in out[i]:
            cs_store.append(j)
        cs.append(summarize_cs(out[i]))

print('Initial CS: ' + str(len(cs)))
print('Initial RS: ' + str(len(rs)))

# OUTPUT INTERMEDIATE RESULTS: ROUND 1
output_intermediate(round, ds, cs, rs, w)

#  BFR CORE
"""
For every file:
    - For every data point:
        - Calculate Mahalanobis distance to each of the ds centroids
            - Assign to the DS if the distance is <2 sqrt(d)
        - If not assigned to any,
            - Calculate Mahalanobis distance to each of the cs centroids
                - Assign to the CS if the distance is <2 sqrt(d)
            - If not assigned to any,
                - Add to RS
    - Cluster RS with a large number of k
        - Add to CS if clusters of more than 1 form, otherwise, keep in RS
    - Merge CS clusters:
        - Calculate Mahalanobis distance between every CS clusters
        - 
"""
# Read in file

previous = []

for file in list_of_files[1:]:
    read_file = input_directory + '/' + file
    round += 1

    read = sc.textFile(read_file).map(lambda x: x.split(',')).map(lambda x: (x[0], tuple([float(i) for i in x[1:]])))

    # Take the previous rs and combine it with the read function now
    previous = [tuple((i, master_points[i])) for i in rs]
    previous += [tuple((i, master_points[i])) for i in cs_store]
    previous = sc.parallelize(previous)

    master_points = read.union(previous).collectAsMap()
    #print(master_points)

    # Calculate Mahalnobis distance to each of the ds centroids for every data point. Assign to ds_temp is less than threshold
    list_of_points = list(read.collectAsMap().keys())

    for i in list_of_points:
        coord = master_points[i]
        threshold = {}
        for j in range(len(ds)):
            summary = (ds[j][3], ds[j][4])
            d = mahalanobis(coord, summary)
            if d < 2*math.sqrt(dim): # if less than threshold
                threshold[j] = d
        # Find min value
        if len(list(threshold.keys())) > 0: # If there are ds within the threshold
            assign = min(threshold, key=threshold.get) # Find minimum of the dictionary
            final_output[int(i)] = assign
            ds[assign] = update_ds(ds[assign], i) # Update ds to fit the next point i
        else: # Means that the point i did not pass any threshold, so look at all CS
            threshold_cs = {}
            if len(cs) == 0: # Edge case when we don't have a cs just yet
                rs.append(i)
            else:
                for j in range(len(cs)):
                    summary = (cs[j][3], cs[j][4])
                    d = mahalanobis(coord, summary)
                    if d < 2 * math.sqrt(dim):
                        threshold_cs[j] = d

                if len(list(threshold_cs.keys())) > 0:
                    assign = min(threshold_cs, key=threshold_cs.get)
                    cs_store.append(i)
                    cs[assign] = update_ds(cs[assign], i)
                else: # Point also not in any CS, so append to RS
                    rs.append(i)

    print('RS before clustering: ' + str(len(rs)))

    # Cluster rs with a large number of k, and see if any new cs forms
    outliers = k_means(2*k, 2, rs)
    rs_temp = []
    for i in outliers:
        if len(outliers[i]) > 1: # If there are more than one points in the cluster, add to cs
            cs.append(summarize_cs(outliers[i]))
            for j in outliers[i]:
                cs_store.append(j)
        else: # If only one point in the cluster, add to rs
            for j in outliers[i]:
                rs_temp.append(j)

    rs = rs_temp
    print('RS after clustering: ' + str(len(rs)))
    print('CS after clustering, and before combining: ' + str(len(cs)))


    # Consider merging the cs clusters that have a mahalanobis distance less than a threshold
    # Create pairs from cs list

    candidate_pairs = []
    paired = []
    for i in range(len(cs)):
        cs1 = cs[i][3] # only take centroid for cs1
        if i not in paired:
            for j in range(i+1, len(cs)):
                if j not in paired:
                    cs2 = (cs[j][3], cs[j][4])
                    d = mahalanobis(cs1, cs2)
                    if d < 2 * math.sqrt(dim):
                        candidate_pairs.append((i, j))
                        paired.append(i)
                        paired.append(j)
                        break

    #print('CS Before combining: ' + str(len(cs)))
    print('Candidate pairs: ' + str(candidate_pairs))
    # For every candidate pair, merge. Update cs, but also pop the candidate_pairs out
    temp = []
    flat = []
    for i in candidate_pairs:
        temp.append(combine_cs(cs[i[0]], cs[i[1]]))
        flat.append(cs[i[0]])
        flat.append(cs[i[1]])

    for i in flat:
        cs.remove(i)

    for j in temp:
        cs.append(j)

    print('CS After combining: ' + str(len(cs)))

    #candidate_pairs = []

    print('Total number in DS' + str(len(final_output))) # this should equal the number of points that's in the discard set
    print('Total number in CS' + str(len(cs_store)))
    # Write out intermediate file
    output_intermediate(round, ds, cs, rs, w)
    print(round)

# MERGE ALL CS AND RS TO CLOSEST DS CLUSTER
# Retrieve all centroids from ds
ds_centroids = [x[3] for x in ds]
for i in cs:
    # take the centroid
    cs_centroid = i[3]
    # find closest centroid in ds
    min = sys.maxsize
    min_cluster = 0
    for j in range(len(ds_centroids)):
        d = compute_distance(cs_centroid, ds_centroids[j])
        if d < min:
            min = d
            min_cluster = j

    # Merge cs to whichever ds_centroid has minimum distance
    updated_ds = merge_to_ds(i, ds[min_cluster])
    ds[min_cluster] = updated_ds

# Merge rs to ds
for i in rs:
    coordinates = master_points[i]
    min = sys.maxsize
    min_cluster = 0
    for j in range(len(ds_centroids)):
        d = compute_distance(coordinates, ds_centroids[j])
        if d < min:
            min = d
            min_cluster = j
    update_ds(ds[min_cluster], coordinates)
    final_output[int(i)] = min_cluster # put all RS points in the final_output

print('Total number in DS after merging RS: ' + str(len(final_output)))

# Iterate through the cs points and see which cluster it's clusest to. Assign to final_output
for i in cs_store:
    coordinates = master_points[i]
    min = sys.maxsize
    min_cluster = 0
    for j in range(len(ds_centroids)):
        d = compute_distance(coordinates, ds_centroids[j])
        if d < min:
            min = d
            min_cluster = j
    final_output[int(i)] = min_cluster

print(len(final_output))


# Order the final_output dictionary
# Write out, then close the file
# order final_output's keys
ordered = sorted(list(final_output.keys()))
dic = {}
w2 = open(outfile2, 'w')
for i in ordered:
    dic[str(i)] = final_output[i]
w2.write(json.dumps(dic))

w.close()
w2.close()

end = time.time()
print(end-start)