import networkx as nx
import itertools as it
import random
import pickle

# build the graph
num_knowledge = 50
num_neighbors = 5
reconnection_rate = 0.1
num_knowledge_per_task = 3
sample_rate = 0.1
G = nx.watts_strogatz_graph(num_knowledge, num_neighbors, reconnection_rate, seed=2023)

# calculate node distances
node_distance_dict = dict(nx.all_pairs_shortest_path_length(G))
f_save = open('/results/knowledge_distance.pkl', 'wb')
pickle.dump(node_distance_dict, f_save)
f_save.close()

# calculate node features
node_betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
node_closeness_centrality = nx.closeness_centrality(G)

node_betweenness_centrality_list = []
node_closeness_centrality_list = []

for i in range(len(node_betweenness_centrality)):
    node_betweenness_centrality_list.append(node_betweenness_centrality[i])
    node_closeness_centrality_list.append(node_closeness_centrality[i])

f_save = open('/results/node_betweenness_centrality.pkl', 'wb')
pickle.dump(node_betweenness_centrality_list, f_save)
f_save.close()
f_save = open('/results/node_closeness_centrality.pkl', 'wb')
pickle.dump(node_closeness_centrality_list, f_save)
f_save.close()

# task value assignment
value_task = []
for i in it.combinations(range(0, num_knowledge), num_knowledge_per_task):
    if random.random() < sample_rate:
        value_task.append(set(i))

f_save = open('/results/task_id_to_knowledge_set.pkl', 'wb')
pickle.dump(value_task, f_save)
f_save.close()

def task_distance(node_distance_dict, task_1, task_2):
    two_to_one = []
    for node_1 in task_1:
        smallest_length = 1e10
        for node_2 in task_2:
            if node_distance_dict[node_1][node_2] < smallest_length:
                smallest_length = node_distance_dict[node_1][node_2]
        two_to_one.append(smallest_length)
    two_to_one = sum(two_to_one) / len(two_to_one)
    
    one_to_two = []
    for node_2 in task_2:
        smallest_length = 1e10
        for node_1 in task_1:
            if node_distance_dict[node_2][node_1] < smallest_length:
                smallest_length = node_distance_dict[node_2][node_1]
        one_to_two.append(smallest_length)
    one_to_two = sum(one_to_two) / len(one_to_two)
    
    return (two_to_one + one_to_two) / 2  

task_distance_dict = {}
for i in range(len(value_task)):
    task_distance_dict[i] = {}
for i in range(len(value_task)):
    for j in range(i+1, len(value_task)):
        result = float(format(task_distance(node_distance_dict, value_task[i], value_task[j]), '.1f'))
        task_distance_dict[i][j] = result
        task_distance_dict[j][i] = result

def source_target_distance(task_distance_dict, source, target):
    smallest_length = 1e10
    smallest_id = -1
    for node in source:
        if task_distance_dict[node][target] < smallest_length:
            smallest_length = task_distance_dict[node][target]
            smallest_id = node
    return smallest_length, smallest_id

def diffuse(task_value, source, task_distance_dict, step, std):
    smallest_length_list = []
    smallest_id_list = []
    target_id_list = []
    for i in range(len(value_task)):
        if i not in source:
            smallest_length, smallest_id = source_target_distance(task_distance_dict, source, i)
            if smallest_length == 1e10 or smallest_id == -1:
                print('error!')
            smallest_length_list.append(smallest_length)
            smallest_id_list.append(smallest_id)
            target_id_list.append(i)

    smallest_length = min(smallest_length_list)
    smallest_id = smallest_id_list[smallest_length_list.index(smallest_length)]
    target_id = target_id_list[smallest_length_list.index(smallest_length)]
    
    value = 0
    for _ in range(int(smallest_length / step)):
        value += random.gauss(0, std)
    
    task_value[target_id] = task_value[smallest_id] + value
    source.append(target_id)

starter = random.randint(0, len(task_distance_dict)-1)
task_value = {starter: 50}
source = [starter]
while len(source) < len(value_task):
    diffuse(task_value, source, task_distance_dict, step=0.1, std=1)

task_id_to_value_list = []
for i in range(len(task_value)):
    task_id_to_value_list.append(task_value[i])

f_save = open('/results/task_id_to_value.pkl', 'wb')
pickle.dump(task_id_to_value_list, f_save)
f_save.close()

# calculate knowledge values
f = open('/results/task_id_to_knowledge_set.pkl', 'rb')
taskid2knowledge = pickle.load(f)
f.close()
f = open('/results/task_id_to_value.pkl', 'rb')
taskid2value = pickle.load(f)
f.close()

node_value = [0] * num_knowledge
for i in range(len(taskid2knowledge)):
    for node in taskid2knowledge[i]:
        node_value[node] += taskid2value[i]

f_save = open('/results/knowledge_value.pkl', 'wb')
pickle.dump(node_value, f_save)
f_save.close()