from pyspark import SparkContext
from itertools import combinations
import time
import sys

def girvan_newman(x):
    #global edge_values
    global edges
    # STEP 1: Calculate number of shortest paths
    master = {0: {x: set()}}
    visited_nodes = {x}
    current_nodes = {x}
    visited_this_level = set()
    next_nodes = set()
    level = 0
    while len(current_nodes) > 0:
        level += 1
        for i in current_nodes:
            dst_nodes = [y for x, y in edges if x == i]
            #print(dst_nodes)
            for j in dst_nodes:
                if j in visited_nodes:
                    pass
                else:
                    next_nodes.add(j)
                    if level not in master:
                        master[level] = {}

                    if j in master[level]:
                        master[level][j].add(i)
                    else:
                        master[level][j] = {i}

                    visited_this_level.add(j)

        current_nodes = next_nodes
        #print(current_nodes)
        #print(current_nodes)
        next_nodes = set()
        visited_nodes = visited_nodes | visited_this_level
        #print(visited_nodes)
        #print(master)
        visited_this_level = set()

    # COUNT SHORTEST PATHS
    max_level = level
    shortest_paths = {0: {x: 1}}
    level = 1
    while level < max_level:
        # print(level)
        shortest_paths[level] = {}
        if level == 1:
            for i in master[level]:
                shortest_paths[level][i] = len(master[level][i])
        else:
            for i in master[level]:
                # print(i)
                count = 0
                for j in master[level][i]:
                    count += shortest_paths[level - 1][j]
                shortest_paths[level][i] = count
            # print(shortest_paths)
        level += 1

    # STEP 2: COUNTING BETWEENNESS
    node_values = {}
    edge_values = []
    #edge_values
    level -= 1  # because the while loop goes one extra time
    lvl = master[level]
    for i in lvl:
        node_values[i] = 1.0

    while level > 0:
        for i in lvl:
            if i not in node_values:
                node_values[i] = 1.0
            den = shortest_paths[level][i]
            for j in lvl[i]:
                r = float(shortest_paths[level - 1][j]) / float(den)
                temp = node_values[i] * r
                if j not in node_values:
                    node_values[j] = temp + 1
                else:
                    node_values[j] += temp
                edge_values.append(((i, j), temp))
                edge_values.append(((j, i), temp))

        level -= 1
        lvl = master[level]

    return edge_values


def find_communities():
    global traversed
    global edges
    current = ''
    for i in traversed:
        if not traversed[i]:
            current = i
            break
    comm_num = 1
    list_of_communities = {}
    while current is not None:
        list_of_communities[comm_num] = set()
        list_of_communities[comm_num].add(current)
        traversed[current] = True

        connected = [y for x, y in edges if x == current]
        for i in connected:
            traversed[i] = True

        while len(connected) > 0:
            check = connected[0]
            connected_buff = [y for x, y in edges if x == check]

            for i in connected_buff:
                if not traversed[i]:
                    connected.append(i)
                    traversed[i] = True

            list_of_communities[comm_num].add(check)
            connected.pop(0)

        comm_num += 1
        current = None
        for i in traversed:
            if not traversed[i]:
                current = i
                break

    return list_of_communities

def calc_modularity():
    global list_of_communities
    global nodes_list
    global m
    global deg
    communities = list(enumerate(list(list_of_communities.values())))
    mod = 0
    for i in nodes_list:
        for j in nodes_list:
            # Check if i and j are in the same community
            if [a for a, lst in communities if i in lst][0] == [b for b, lst in communities if j in lst][0]:
                if (i, j) not in edges:
                    mod += 0 - float(deg[i] * deg[j]) / float(2 * m)
                else:
                    mod += 1 - float(deg[i] * deg[j]) / float(2 * m)

    return mod / (2 * m)

def calculate_degree():
    global nodes_list
    global edges
    deg = {}
    for i in nodes_list:
        num = [y for x, y in edges if x == i]
        deg[i] = len(num)

    return deg

# ------ CODE STARTS -------
# ead input file
filter_threshold = int(sys.argv[1])
input_file = sys.argv[2]
betweenness_output = sys.argv[3]
community_output = sys.argv[4]

"""filter_threshold = 7
input_file = 'ub_sample_data.csv'
betweenness_output = 'output.txt'
community_output = 'test.txt'"""

start = time.time()
sc = SparkContext('local[*]', 'task2')

lines = sc.textFile(input_file).map(lambda x: x.split(',')).map(lambda x: (x[0], x[1]))

# Find nodes and edges with at least 7 corated businessed among the user pairs
user_business = lines.groupByKey().map(lambda x: (x[0], set(x[1]))).collectAsMap()
user_list = user_business.keys() # 3375 unique users

user_pairs = [] # 498 pairs
for x, y in combinations(user_list, 2):
    if len(user_business[x] & user_business[y]) >= filter_threshold:
        user_pairs.append((x, y))

nodes = sc.parallelize(user_pairs).flatMap(lambda x: list(x)).distinct() # 222 distinct users

edges = sc.parallelize(user_pairs)
m = edges.count()
edges2 = edges.map(lambda x: (x[1], x[0]))
edges_rdd = edges.union(edges2)

#list_of_mod = []

# -- CONSTANT --
nodes_list = nodes.collect()

# -- CHANGES EVERY CUT --
edges = edges_rdd.collect()
traversed = nodes.map(lambda x: (x, False)).collectAsMap() # to recalibrate the community detection
list_of_communities = find_communities()

# -- ONLY CHANGED IF COMMUNITIES CHANGE --> FOR MODULARITY --
deg = calculate_degree()
comm_num = len(list_of_communities.keys())
#print(calc_modularity())


timeout = time.time() + 200
max_mod = -1
max_communities = {}
i = 0
while len(edges) > 0:
    changed = False
    counter = 0
    while not changed:
        # Keep calculating betweenness with g-n and cutting edges
        calc = nodes.flatMap(girvan_newman).reduceByKey(lambda x, y: x + y).map(lambda x: (x[1] / 2, tuple(sorted(x[0])))).distinct() \
            .groupByKey().map(lambda x: (x[0], sorted(list(x[1])))).sortByKey(ascending=False)

        if i == 0 and counter == 0:
            with open(betweenness_output, 'w') as w:
                for j in calc.collectAsMap():
                    for k in calc.collectAsMap()[j]:
                        w.write(str(k) + ', ' + str(j) + '\n')
                w.close()

        # Make Cut by updating edges
        highest = calc.map(lambda x: x[1]).take(1)[0]

        edges_rdd = sc.parallelize(edges)
        for j in highest:
            edges_rdd = edges_rdd.filter(lambda x: x != j).filter(lambda x: x != (j[1], j[0]))
        edges = edges_rdd.collect()

        traversed = nodes.map(lambda x: (x, False)).collectAsMap()
        list_of_communities = find_communities()

        counter += 1
        if len(list_of_communities.keys()) > comm_num:
            changed = True
            comm_num = len(list_of_communities.keys())

    mod = calc_modularity()
    if mod > max_mod:
        max_mod = mod
        max_communities = list_of_communities

    if time.time() > timeout:
        break

    """if max_mod-mod > 0.3:
        break"""
    i += 1
print(max_mod)
print(len(max_communities.keys()))


communities = sc.parallelize(list(max_communities.values())).map(lambda x: (len(x), sorted(list(x)))).groupByKey().\
    map(lambda x: (x[0], sorted(list(x[1])))).sortByKey().collectAsMap()

with open(community_output, 'w') as w:
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