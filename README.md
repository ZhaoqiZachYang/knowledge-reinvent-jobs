# Evolutionary modeling reveals that value-oriented knowledge creation behaviors reinvent jobs

This repository includes codes and data used in the paper "Evolutionary modeling reveals that value-oriented knowledge creation behaviors reinvent jobs", which has been accepted in principle by Humanities and Social Sciences Communications (Nature HSSCOMMS).

## Codes

First, run environment_setup.py to build up the environment.
You can change multiple simulation variables here.

Then, run simulation.py for simulation.
The results will be shown in experiment_name.txt.

Finally, run real_data.py to analyze the real-world data set.
You can choose one of the four mechanisms here.

## Real-world Data

data_1.txt: Real-world mechanism proportions presented in the main article.

data_2.txt: Real-world mechanism proportions presented in the supplementary materials.

jd_knowledge_count.txt: The number of knowledge (skill keywords) appearing in job descriptions.

company_region.pkl: A Python dictionary mapping company names to their respective regions.

1: East Coast, 2: Northeast Region, 3: Central Region, 4: West Region

company_industry.pkl: A Python dictionary mapping company names to their respective industries.

1: Agriculture, Forestry, Animal Husbandry, and Fishery,
2: Mining,
3: Manufacturing,
4: Electricity, Heat, Gas, and Water Production and Supply,
5: Construction,
6: Wholesale and Retail,
7: Transportation, Warehousing, and Postal Services,
8: Accommodation and Catering,
9: Information Transmission, Software, and Information Technology Services,
10: Finance,
11: Real Estate,
12: Leasing and Business Services,
13: Scientific Research and Technical Services,
14: Water Conservancy, Environment, and Public Facilities Management,
15: Resident Services, Repair, and Other Services,
16: Education,
17: Health and Social Work,
18: Culture, Sports, and Entertainment.

Due to privacy reasons, we anonymize all company names.


## Simulated Data

main_results.txt: Main results of our simulation.

Due to the randomness in our simulation algorithm, results may vary slightly depending on the random seed used.

