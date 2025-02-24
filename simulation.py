import mesa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import random
import pickle
import copy
import itertools as it
from datetime import datetime
import argparse


class OrganizationAgent(mesa.Agent):
    """An agent"""

    def __init__(self, unique_id, model, num_initial_task, knowledge_width):
        super().__init__(unique_id, model)
        self.learned_knowledge_set = set()
        self.ranked_tasks = []
        self.log_file = model.log_file
        self.wealth = 0
        
        center_knowledge = random.randint(0, self.model.num_knowledge-1)
        low_boundary = (center_knowledge - knowledge_width) % self.model.num_knowledge
        high_boundary = (center_knowledge + knowledge_width) % self.model.num_knowledge
        if knowledge_width >= self.model.num_knowledge / 2:
            knowledge_range = list(range(self.model.num_knowledge))
        elif low_boundary < high_boundary:
            knowledge_range = list(range(low_boundary, high_boundary))
        else:
            knowledge_range = list(range(low_boundary, self.model.num_knowledge)) + list(range(high_boundary))
        
        ready_pre_tasks = []
        
        for pre_task in self.model.taskid2knowledge:
            flag = 0
            for knowledge in pre_task:
                if knowledge not in knowledge_range:
                    flag = 1
                    break
            if flag == 1:
                continue
            else:
                ready_pre_tasks.append(pre_task) 
               
        i = 0
        while(i < num_initial_task):
            flag = 0
            pre_task = random.sample(ready_pre_tasks, 1)
            for knowledge_id in pre_task[0]:
                if not self.check_knowledge_capacity(knowledge_id):
                    flag = 1
                    break
            if flag == 1:
                continue
            
            ranked_task = list(pre_task[0])
            random.shuffle(ranked_task)
            
            if ranked_task in self.ranked_tasks:
                continue
            for j in range(len(ranked_task)):
                self.model.knowledge_usage[ranked_task[j]] += self.model.num_knowledge_per_task - j
            self.ranked_tasks.append(ranked_task)
            self.learned_knowledge_set = self.learned_knowledge_set.union(pre_task[0])
            
            i += 1
        
        for knowledge_id in self.learned_knowledge_set:
            self.model.knowledge_org_num[knowledge_id] += 1
            
        print('Agent ', self.unique_id, ': Initial Tasks: ', self.ranked_tasks, file = self.log_file)
        print('*'*10, file = self.log_file)
        
            
    def check_knowledge_capacity(self, knowledge_id):
        if self.model.knowledge_usage[knowledge_id] == self.model.knowledge_capacity:
            return False
        elif self.model.knowledge_usage[knowledge_id] > self.model.knowledge_capacity:
            print('error!')
            return False
        else:
            return True  
    
    def task_evolution(self):
        profit_list = [0]
        info_list = ['do nothing']
        not_transfer_flag = 0
        
        # task transfer
        for agent in self.model.schedule.agents:
            for transfer_ranked_task in agent.ranked_tasks:
                if not_transfer_flag == 1 and transfer_ranked_task in self.ranked_tasks:
                    continue
                after_transfer_ranked_tasks = copy.deepcopy(self.ranked_tasks)
                new_learned_knowledge_set = copy.deepcopy(self.learned_knowledge_set)
                if transfer_ranked_task not in self.ranked_tasks:
                    after_transfer_ranked_tasks.append(transfer_ranked_task)
                    cost_4 = 0
                    for knowledge in transfer_ranked_task:
                        if knowledge not in new_learned_knowledge_set:
                            cost_4 += self.model.competition_cost(knowledge) + \
                                      self.model.knowledge_search_cost(new_learned_knowledge_set, knowledge)
                    cost_4 = cost_4 * self.model.task_transfer_cost(agent.ranked_tasks, transfer_ranked_task)
                    new_learned_knowledge_set = new_learned_knowledge_set.union(set(transfer_ranked_task))
                    new_value_3 = self.model.ranked_task_value(transfer_ranked_task)
                else:
                    not_transfer_flag = 1
                    cost_4 = 0
                    new_value_3 = 0
                
                # knowledge permutation
                for task in after_transfer_ranked_tasks:
                    old_value_1 = self.model.ranked_task_value(task)
                    new_ranked_tasks = copy.deepcopy(after_transfer_ranked_tasks)
                    task_index = new_ranked_tasks.index(task)
                    for new_task_1 in list(it.permutations(task, self.model.num_knowledge_per_task)) + [[]]:
                        new_task_1 = list(new_task_1)
                        new_knowledge_usage = copy.deepcopy(self.model.knowledge_usage)
                        for j in range(len(task)):
                            new_knowledge_usage[task[j]] -= self.model.num_knowledge_per_task - j
                        flag = 0
                        new_value_1 = self.model.ranked_task_value(new_task_1)
                        if new_task_1 != task:
                            cost_1 = self.model.knowledge_coupling_cost(new_task_1)
                        else:
                            cost_1 = 0
                        for i in range(len(new_task_1)):
                            new_knowledge_usage[new_task_1[i]] += self.model.num_knowledge_per_task - i
                            if new_knowledge_usage[new_task_1[i]] > self.model.knowledge_capacity:
                                flag = 1
                                break
                        if flag == 1:
                            continue
                        new_ranked_tasks[task_index] = new_task_1
                        
                        # knowledge search
                        for knowledge in range(self.model.num_knowledge):
                            for p in range(len(new_ranked_tasks)):
                                for q in range(len(new_ranked_tasks[p])):
                                    new_knowledge_usage[new_ranked_tasks[p][q]] -= self.model.num_knowledge_per_task - q 
                                    if new_knowledge_usage[knowledge] + self.model.num_knowledge_per_task - q <= \
                                       self.model.knowledge_capacity:
                                        old_value_2 = self.model.ranked_task_value(new_ranked_tasks[p])
                                        new_task_2 = copy.deepcopy(new_ranked_tasks[p])
                                        new_task_2[q] = knowledge
                                        new_value_2 = self.model.ranked_task_value(new_task_2)
                                        if new_value_2 == 0:
                                            continue
                                        if new_task_2 == new_ranked_tasks[p]:
                                            cost_2 = 0
                                        else:
                                            cost_2 = self.model.knowledge_coupling_cost(new_task_2)
                                        if new_ranked_tasks[p] == new_task_1:
                                            cost_1 = 0
                                        if knowledge in new_learned_knowledge_set:
                                            cost_3 = 0
                                        else:
                                            cost_3 = self.model.knowledge_search_cost(new_learned_knowledge_set, knowledge) \
                                                    + self.model.competition_cost(knowledge)
                                        profit = new_value_3 + new_value_1 + new_value_2 - cost_1 - cost_2 - \
                                                 old_value_1 - old_value_2 - cost_3 - cost_4
                                        profit_list.append(profit)
                                        
                                        info_list.append((copy.deepcopy(transfer_ranked_task), 
                                                          transfer_ranked_task not in self.ranked_tasks,
                                                          copy.deepcopy(task), 
                                                          copy.deepcopy(new_task_1), 
                                                          copy.deepcopy(new_ranked_tasks[p]), 
                                                          copy.deepcopy(new_task_2), 
                                                          knowledge not in new_learned_knowledge_set,
                                                          knowledge, agent))
                                    new_knowledge_usage[new_ranked_tasks[p][q]] += self.model.num_knowledge_per_task - q


        max_profit = max(profit_list)
        max_info = info_list[profit_list.index(max_profit)]
        return max_profit, max_info     
    
    

    def step(self):
        
        profit, info = self.task_evolution()
        self.wealth += profit
        
        if info == 'do nothing':
            print('Agent ', self.unique_id, ': Do Nothing.', file = self.log_file)
            print('*'*10, file = self.log_file)
        
        else:
            
            transfered_task, is_transfered, pre_permutation_task, after_permutation_task, \
            pre_search_task, after_search_task, is_search, searched_knowledge, target_agent = info
            flag = 0
            
            if is_transfered:
                flag = 1
                for knowledge in transfered_task:
                    if knowledge not in self.learned_knowledge_set:
                        self.model.knowledge_org_num[knowledge] += 1
                self.learned_knowledge_set = self.learned_knowledge_set.union(set(transfered_task))
                self.ranked_tasks.append(transfered_task)
                target_agent.ranked_tasks.remove(transfered_task)
                print('Agent ', self.unique_id, ': Task Transfered: ', 
                      transfered_task, ' from Agent ', target_agent.unique_id, file = self.log_file)
            
            if after_permutation_task != pre_permutation_task:
                flag = 1
                for i in range(len(pre_permutation_task)):
                    self.model.knowledge_usage[pre_permutation_task[i]] -= self.model.num_knowledge_per_task - i
                for i in range(len(after_permutation_task)):
                    self.model.knowledge_usage[after_permutation_task[i]] += self.model.num_knowledge_per_task - i
                self.ranked_tasks[self.ranked_tasks.index(pre_permutation_task)] = after_permutation_task
                print('Agent ', self.unique_id, ': Task Permutation from: ', 
                      pre_permutation_task, ' to ', after_permutation_task, file = self.log_file)
            
            if pre_search_task != after_search_task and is_search:
                flag = 1
                for i in range(len(pre_search_task)):
                    self.model.knowledge_usage[pre_search_task[i]] -= self.model.num_knowledge_per_task - i
                for i in range(len(after_search_task)):
                    self.model.knowledge_usage[after_search_task[i]] += self.model.num_knowledge_per_task - i
                self.ranked_tasks[self.ranked_tasks.index(pre_search_task)] = after_search_task
                self.learned_knowledge_set = self.learned_knowledge_set.union(set([searched_knowledge]))
                self.model.knowledge_org_num[searched_knowledge] += 1
                print('Agent ', self.unique_id, ': Knowledge Search from: ', 
                      pre_search_task, ' to ', after_search_task, file = self.log_file)
           
            elif pre_search_task != after_search_task:
                flag = 1
                for i in range(len(pre_search_task)):
                    self.model.knowledge_usage[pre_search_task[i]] -= self.model.num_knowledge_per_task - i
                for i in range(len(after_search_task)):
                    self.model.knowledge_usage[after_search_task[i]] += self.model.num_knowledge_per_task - i
                self.ranked_tasks[self.ranked_tasks.index(pre_search_task)] = after_search_task
                print('Agent ', self.unique_id, ': Knowledge Reuse from: ', 
                      pre_search_task, ' to ', after_search_task, file = self.log_file)
                
            if flag == 0:
                print('Agent ', self.unique_id, ': Do Nothing.', file = self.log_file)
                print('*'*10, file = self.log_file)
                return
            
            print('Agent ', self.unique_id, ': Profit: ', profit, file = self.log_file)
            print('*'*10, file = self.log_file)



class LaborMarketModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, num_agents, num_knowledge, num_neignbors, p_reconnect, seed, 
                 path_taskid2value, path_taskid2knowledge, 
                 path_knowledge_distance,knowledge_capacity, num_initial_task,
                 num_knowledge_per_task, knowledge_width):
        
        self.num_agents = num_agents
        self.num_knowledge = num_knowledge
        self.knowledge_network = nx.watts_strogatz_graph(num_knowledge, num_neignbors, p_reconnect, seed)
        self.knowledge_capacity = knowledge_capacity
        self.knowledge_usage = [0 for _ in range(self.num_knowledge)]
        self.knowledge_org_num = [0 for _ in range(self.num_knowledge)]
        self.num_knowledge_per_task = num_knowledge_per_task
        self.num_initial_task = num_initial_task
        
        self.experiment_name = 'an'+str(num_agents)+'_'+'kn'+str(num_knowledge)+'_'+'kc'+str(knowledge_capacity)+'_'+\
                               'it'+str(num_initial_task)+'_'+'npt'+str(num_knowledge_per_task)+'_'+str(datetime.now())
        self.log_file = open('/results/'+self.experiment_name+'.txt','w')
        
        f_read = open(path_knowledge_distance, 'rb')
        self.knowledge_distance = pickle.load(f_read)
        f_read.close()
        f_read = open(path_taskid2value, 'rb')
        self.taskid2value = pickle.load(f_read)
        f_read.close()
        f_read = open(path_taskid2knowledge, 'rb')
        self.taskid2knowledge = pickle.load(f_read)
        f_read.close()
        
        self.schedule = mesa.time.RandomActivation(self)
        
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            a = OrganizationAgent(i, self, num_initial_task, knowledge_width)
            self.schedule.add(a)
        
        self.datacollector = mesa.DataCollector(agent_reporters={"Wealth": "wealth"})
 
        
    def ranked_task_value(self, knowledge_rank):
        node_centrality = []
        
        if set(knowledge_rank) in self.taskid2knowledge:
            task_id = self.taskid2knowledge.index(set(knowledge_rank))
            value = self.taskid2value[task_id]
            for node_1 in knowledge_rank:
                temp = 0
                for node_2 in knowledge_rank:
                    temp += self.knowledge_distance[node_1][node_2]
                node_centrality.append(1 / temp)
            temp = 0
            for i in range(len(node_centrality)):
                temp += node_centrality[i] * (len(node_centrality) + 1 - i) * 2 / (len(node_centrality) + 1)
            return value * temp / sum(node_centrality)
        else:
            return 0  
        
    def knowledge_search_cost(self, learned_knowledge_set, knowledge_id):
        distance_list = []
        for node in learned_knowledge_set:
            distance_list.append(self.knowledge_distance[node][knowledge_id])
        return min(distance_list)
          
    def knowledge_coupling_cost(self, task_knowledge_set):
        if len(task_knowledge_set) == 0:
            return 0
        temp = 0
        for node_1 in task_knowledge_set:
            for node_2 in task_knowledge_set:
                if node_1 != node_2:
                    temp += self.knowledge_distance[node_1][node_2]
        return temp / len(task_knowledge_set) / (len(task_knowledge_set) - 1)
    
    def task_transfer_cost(self, target_agent_tasks, target_task):
        temp = 0
        for task in target_agent_tasks:
            temp += self.ranked_task_value(task)
        return self.ranked_task_value(target_task) / temp * 500
    
    def competition_cost(self, knowledge_id):
        return self.knowledge_org_num[knowledge_id]
        
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()



if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='manual to this script')
    
    parser.add_argument('--num-agents', type=int, default = 10)
    parser.add_argument('--num-knowledge', type=int, default=50)
    parser.add_argument('--num-neignbors', type=int, default = 5)
    parser.add_argument('--p-reconnect', type=float, default=0.1) 
    parser.add_argument('--seed', type=int, default = 2023)
    parser.add_argument('--num-knowledge-per-task', type=int, default=3)
    parser.add_argument('--knowledge-width', type=int, default = 7)
    parser.add_argument('--knowledge-capacity', type=int, default = 50)
    parser.add_argument('--num-initial-task', type=int, default = 3)
    parser.add_argument('--path-taskid2value', type=str, default = '/results/task_id_to_value.pkl')
    parser.add_argument('--path-taskid2knowledge', type=str, default='/results/task_id_to_knowledge_set.pkl')
    parser.add_argument('--path-knowledge-distance', type=str, default = '/results/knowledge_distance.pkl')
    parser.add_argument('--path-closeness-centrality', type=str, default = '/results/node_closeness_centrality.pkl')
    parser.add_argument('--path-betweenness-centrality', type=str, default='/results/node_betweenness_centrality.pkl')
    parser.add_argument('--iterations', type=int, default = 1)
    parser.add_argument('--max-steps', type=int, default = 10)
    parser.add_argument('--number-processes', type=int, default = 4)
    
    args = parser.parse_args()
    
    model = LaborMarketModel(num_agents=args.num_agents, num_knowledge=args.num_knowledge,
                             num_neignbors=args.num_neignbors, p_reconnect=args.p_reconnect, seed=args.seed, 
                             path_taskid2value=args.path_taskid2value,
                             path_taskid2knowledge=args.path_taskid2knowledge, 
                             path_knowledge_distance=args.path_knowledge_distance,
                             knowledge_capacity=args.knowledge_capacity, 
                             num_initial_task=args.num_initial_task,
                             num_knowledge_per_task = args.num_knowledge_per_task, 
                             knowledge_width=args.knowledge_width)
    
    fig = plt.figure(figsize=(30, 20))
    pos = nx.circular_layout(model.knowledge_network)

    for p in range(model.num_agents):
        plt.subplot(4,5,p+1)
        nx.draw(model.knowledge_network, pos=pos, node_size=10, node_color='#DCDCDC', edge_color='#DCDCDC')
        j = 0
        for task in model.schedule.agents[p].ranked_tasks:
            task_directed_G = nx.DiGraph()
            task_directed_G.add_nodes_from(task)
            for i in range(len(task)):
                if i != 0:  
                    task_directed_G.add_edge(task[i-1], task[i])
            nx.draw(task_directed_G, pos=pos, node_color='blue', edge_color='blue', 
                    style = 'dotted', node_size=30)
            j += 1       
        
        plt.savefig('/results/'+model.experiment_name+'_initialization.png')
  
    
    node_degree = [0] * model.num_knowledge
    node_value = [0] * model.num_knowledge
    for i in range(len(model.taskid2knowledge)):
        for node in model.taskid2knowledge[i]:
            node_value[node] += model.taskid2value[i]
            node_degree[node] += 1
    
    initial_knowledge_usage = copy.deepcopy(model.knowledge_usage)
    initial_knowledge_org_num = copy.deepcopy(model.knowledge_org_num)
    
    for i in range(args.max_steps):
        model.step()
        
    ending_knowledge_usage = model.knowledge_usage
    ending_knowledge_org_num = model.knowledge_org_num
    
    dis_knowledge_usage = []
    dis_knowledge_org_num = []
    
    for i in range(len(ending_knowledge_usage)):
        dis_knowledge_usage.append(ending_knowledge_usage[i] - initial_knowledge_usage[i])
        dis_knowledge_org_num.append(ending_knowledge_org_num[i] - initial_knowledge_org_num[i])
        
    f_read = open(args.path_closeness_centrality, 'rb')
    closeness_centrality = pickle.load(f_read)
    f_read.close()
    f_read = open(args.path_betweenness_centrality, 'rb')
    betweenness_centrality = pickle.load(f_read)
    f_read.close()
        
    fig = plt.figure(figsize=(30, 20))
    pos = nx.circular_layout(model.knowledge_network)

    for p in range(model.num_agents):
        plt.subplot(4,5,p+1)
        nx.draw(model.knowledge_network, pos=pos, node_size=10, node_color='#DCDCDC', edge_color='#DCDCDC')
        j = 0
        drawed_node = []
        for task in model.schedule.agents[p].ranked_tasks:
            task_directed_G = nx.DiGraph()
            task_directed_G.add_nodes_from(task)
            for i in range(len(task)):
                if i != 0:  
                    task_directed_G.add_edge(task[i-1], task[i])
                drawed_node.append(task[i])
            nx.draw(task_directed_G, pos=pos, node_color='blue', edge_color='blue', 
                    style = 'dotted', node_size=30)
            j += 1
         
        print('Agent ', model.schedule.agents[p].unique_id, ': Ending Tasks: ', 
                      model.schedule.agents[p].ranked_tasks, file = model.log_file)
        print('Agent ', model.schedule.agents[p].unique_id, ': Ending Knowledge Set: ', 
                      model.schedule.agents[p].learned_knowledge_set, file = model.log_file)
        print('*'*10, file = model.log_file)
        
        unused_node = []
        for node in model.schedule.agents[p].learned_knowledge_set:
            if node not in drawed_node:
                unused_node.append(node)
        nx.draw(model.knowledge_network.subgraph(unused_node), pos=pos, node_color='green', 
                edge_color='#DCDCDC', node_size=30)

    plt.savefig('/results/'+model.experiment_name+'_ending.png')
    model.log_file.close()
    
    
    df = pd.DataFrame(list(zip(betweenness_centrality, closeness_centrality, 
                               node_degree, node_value, 
                               initial_knowledge_usage, initial_knowledge_org_num,
                               ending_knowledge_usage, ending_knowledge_org_num,
                               dis_knowledge_usage, dis_knowledge_org_num)), 
                               columns =['betweenness_centrality', 'closeness_centrality',
                                         'node_degree', 'node_value',
                                         'initial_knowledge_usage', 'initial_knowledge_org_num',
                                         'ending_knowledge_usage', 'ending_knowledge_org_num',
                                         'dis_knowledge_usage' ,'dis_knowledge_org_num'])
    
    df.to_csv('/results/'+model.experiment_name+'_nodel_feature.csv')