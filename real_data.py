import numpy as np
import matplotlib.pyplot as plt

# real data in the main paper
with open('/data/data_sample_1.txt', 'r') as file:
    lines = file.readlines()
'''
# supplementary real data in the appendix
with open('data_sample_2.txt', 'r') as file:
    lines = file.readlines()
'''

# the mechenism you want to analyze
data_type = 'knowledge_reuse'
#data_type = 'knowledge_search'
#data_type = 'knowledge_permutation'
#data_type = 'task_transfer'

company_names = []
company_data = {}
current_company = ""
current_time_series = {}

for line in lines:
    
    line = line.strip()
    
    if line.startswith("task_transfer") or line.startswith("knowledge_reuse") or line.startswith("knowledge_permutation") or line.startswith("knowledge_search"):
        
        time_series_type = line.split()[0]
        
        time_series = eval(line.replace(time_series_type, ''))
        current_time_series[time_series_type] = time_series
    
    elif line:
        if current_company:
            company_data[current_company] = current_time_series
        current_company = line
        company_names.append(current_company)
        current_time_series = {}

feature_data = {}
current_company = ""
current_time_series = {}
company_weights = {}

with open('/data/jd_knowledge_count.txt', 'r') as file:
    lines = file.readlines()

for line in lines:

    line = line.strip()
    
    if line.startswith("knowledge_count") or line.startswith("jd_count"):
        
        time_series_type = line.split()[0]
        
        time_series = eval(line.replace(time_series_type, '')) 
        current_time_series[time_series_type] = time_series
    
    elif line:
        if current_company:
            feature_data[current_company] = current_time_series
        current_company = line
        current_time_series = {}
    
company_weights = {}
for company_name in feature_data.keys():
    company_weights[company_name] = sum(feature_data[company_name]['jd_count'])

for company_name in feature_data.keys():
    if company_name not in company_weights:
        company_weights[company_name] = 0.0

# calculate VTR
feature_3_data = {}
for company_name, features in feature_data.items():
    factorial_feature_1 = [np.math.factorial(i) for i in np.array(features['knowledge_count']).astype(int)]
    feature_3 = np.array(features['jd_count']) / np.array(factorial_feature_1)
    feature_3_data[company_name] = feature_3

company_feature_tuples = [(company_name, np.mean(feature_3_data[company_name])) for company_name in feature_3_data]

# classify companies into three groups according to VTR
sorted_companies = sorted(company_feature_tuples, key=lambda x: x[1])
num_companies = len(sorted_companies)
group_size = num_companies // 7

group_1_companies = [company[0] for company in sorted_companies[:group_size]]
group_2_companies = [company[0] for company in sorted_companies[3*group_size:4 * group_size]]
group_3_companies = [company[0] for company in sorted_companies[6 * group_size:]]

def calculate_weighted_average(indices, data, weights):
    weighted_averages = []
    
    max_length = max(len(series[data_type]) for series in data.values())
    for t in range(max_length):
        weighted_average = 0

        for company_name in indices:
            if company_name in data and t < len(data[company_name][data_type]):
                weighted_series = data[company_name][data_type][t] * weights[company_name]
                weighted_average += weighted_series

        weighted_average /= np.sum([weights[i] for i in indices])
        weighted_averages.append(weighted_average)
    
    return weighted_averages

group_1_weighted_averages = calculate_weighted_average(group_1_companies, company_data, company_weights)
group_2_weighted_averages = calculate_weighted_average(group_2_companies, company_data, company_weights)
group_3_weighted_averages = calculate_weighted_average(group_3_companies, company_data, company_weights)
all_group_weighted_averages = calculate_weighted_average(feature_data.keys(), company_data, company_weights)


plt.figure(figsize=(12, 6))
plt.plot(group_1_weighted_averages, label='Group 1')
plt.plot(group_2_weighted_averages, label='Group 2')
plt.plot(group_3_weighted_averages, label='Group 3')
plt.title('Weighted Average of %s Over Time for Three Groups' % data_type)
plt.xlabel('Time')
plt.ylabel('Weighted Average')
plt.legend()
plt.savefig('/results/'+'%s_over_time.png' % data_type)
