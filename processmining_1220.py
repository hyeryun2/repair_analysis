from pm4py.objects.conversion.log import converter as log_converter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


repair = pd.read_csv('repair2.csv')
repair2 = repair.copy()
repair.info()




# =============================================================================
# # #데이터준비
# =============================================================================
repair['timestamp'] = repair2['date'] +' ' + repair['time']
repair['timestamp'] = repair['timestamp'].str.replace('1970','2020')
repair['timestamp']  = pd.to_datetime(repair['timestamp'])
repair['timestamp'].isna().sum()

target = repair.loc[repair.timestamp >='2020-01-01']

# target.timestamp.isna().sum()
# target.groupby('caseID').count()
target5 = repair.loc[repair.timestamp >='2022-01-01']
target5.groupby('caseID').count()

len(repair.loc[repair['taskID'] == 'ExternRepair' ])
target = target.sort_values(['caseID','timestamp'])

target.loc[target['taskID'] == 'ExternRepair', 'eventtype' ] ='complete'

target2 = target.copy()
target2 = target2[target2.eventtype != 'start']


target.sort_values('caseID', ascending = False)

1000-927
439+370+118

# =============================================================================
# externRepair개수
# =============================================================================
target3 = target2[target2.eventtype != 'start']
target3 = target3.groupby('caseID').count()
max(target3.originator)
max(target3.eventtype)

expair = target3.loc[target3['RepairType'] ==0]

ex_total = pd.merge(target2,expair, how = 'left', on = 'caseID')
ex_total = ex_total.dropna(subset = ['taskID_y'])

ex_total = ex_total[['caseID','taskID_x','eventtype_x','timestamp_x']]
ex_total.sort_values('caseID')
ex_total = ex_total.drop('eventtype_x', axis = 1)

colname = ['case:concept:name', 'concept:name', 'time:timestamp']
ex_total.columns = colname

ex_total['case:concept:name'] = ex_total['case:concept:name'].astype(str)

#로그변환
log = log_converter.apply(ex_total, variant=log_converter.Variants.TO_DATA_FRAME) 

from pm4py.algo.discovery.dfg import algorithm as dfg_algorithm

dfg = dfg_algorithm.apply(log)
net, im, fm = dfg_mining_factory.apply(dfg)
gviz = dfg_vis_factory.apply(dfg, log = log, variant='performance')
dfg_vis_factory.view(gviz)



repair.info()

# =============================================================================
# Repair Type_E 
# =============================================================================


target
tt = target.dropna(subset = ['RepairType'])
tte = tt.loc[tt['RepairType'] == 'E']

tte_total = pd.merge(target2,tte, how = 'left', on = 'caseID')
tte_total = tte_total.dropna(subset = ['timestamp_y'])

tte_total[]
tte_total = tte_total[['caseID','taskID_x','eventtype_x','timestamp_x']]
tte_total.sort_values('caseID')
tte_total = tte_total.drop('eventtype_x', axis = 1)

colname = ['case:concept:name', 'concept:name', 'time:timestamp']
tte_total.columns = colname

tte_total['case:concept:name'] = tte_total['case:concept:name'].astype(str)

#로그변환
log = log_converter.apply(tte_total, variant=log_converter.Variants.TO_DATA_FRAME) 

from pm4py.algo.discovery.dfg import algorithm as dfg_algorithm

dfg = dfg_algorithm.apply(log)
net, im, fm = dfg_mining_factory.apply(dfg)
gviz = dfg_vis_factory.apply(dfg, log = log, variant='performance')
dfg_vis_factory.view(gviz)


# =============================================================================
# Repair Type_P
# =============================================================================
target
tt = target.dropna(subset = ['RepairType'])
ttp = tt.loc[tt['RepairType'] == 'P']

ttp_total = pd.merge(target2,ttp, how = 'left', on = 'caseID')
ttp_total = ttp_total.dropna(subset = ['timestamp_y'])

ttp_total = ttp_total[['caseID','taskID_x','eventtype_x','timestamp_x']]
ttp_total.sort_values('caseID')
ttp_total = ttp_total.drop('eventtype_x', axis = 1)

colname = ['case:concept:name', 'concept:name', 'time:timestamp']
ttp_total.columns = colname

ttp_total['case:concept:name'] = ttp_total['case:concept:name'].astype(str)

#로그변환
log = log_converter.apply(ttp_total, variant=log_converter.Variants.TO_DATA_FRAME) 

from pm4py.algo.discovery.dfg import algorithm as dfg_algorithm

dfg = dfg_algorithm.apply(log)
net, im, fm = dfg_mining_factory.apply(dfg)
gviz = dfg_vis_factory.apply(dfg, log = log, variant='performance')
dfg_vis_factory.view(gviz)


# =============================================================================
# Repair Type_P
# =============================================================================
target
tt = target.dropna(subset = ['RepairType'])
ttb = tt.loc[tt['RepairType'] == 'B']

ttb_total = pd.merge(target2,ttb, how = 'left', on = 'caseID')
ttb_total = ttb_total.dropna(subset = ['timestamp_y'])

ttb_total = ttb_total[['caseID','taskID_x','eventtype_x','timestamp_x']]
ttb_total.sort_values('caseID')
ttb_total = ttb_total.drop('eventtype_x', axis = 1)

colname = ['case:concept:name', 'concept:name', 'time:timestamp']
ttb_total.columns = colname

ttb_total['case:concept:name'] = ttb_total['case:concept:name'].astype(str)

#로그변환
log = log_converter.apply(ttb_total, variant=log_converter.Variants.TO_DATA_FRAME) 

from pm4py.algo.discovery.dfg import algorithm as dfg_algorithm

dfg = dfg_algorithm.apply(log)
net, im, fm = dfg_mining_factory.apply(dfg)
gviz = dfg_vis_factory.apply(dfg, log = log, variant='performance')
dfg_vis_factory.view(gviz)


# =============================================================================
# NA처리
# =============================================================================
target.isna().sum()
target[target['taskID'].isna()]
target[target['eventtype'].isna()]

target = target.dropna(subset = ['taskID','originator','eventtype'])

target.columns

target = target.drop(['date'], axis = 1)
target = target.drop(['time'], axis = 1)
target2 = target.sort_values(['caseID', 'timestamp'])
target2 = target2[target2.eventtype != 'start']


# =============================================================================
# 업무별 프로세스 확인
# ExternRepair, InformClientWrongPlace 제외 대부분 1 번 이상 진행
# =============================================================================
target4 = target2[['caseID','taskID','originator','eventtype']]

#프로세스 가장 많은 
taskfreq = target4.groupby('taskID').count()
taskfreq = taskfreq['caseID']
taskfreq = pd.DataFrame(taskfreq)
taskfreq.reset_index(inplace = True)
taskfreq.sort_values('caseID', ascending = False, inplace= True)
sns.barplot(data = taskfreq, x='caseID',y='taskID')
plt.title('Frequency of Work Tasks')
plt.show

task = pd.get_dummies(target4.taskID)
target4 = pd.concat([target4, task], axis = 1)
target4 = target4.drop('taskID', axis = 1)

casegroup = target4.groupby('caseID').sum()

plt.figure(figsize=(15,10))
ax = sns.heatmap(casegroup,cmap='Greens',vmin=0, vmax=2)
plt.title('Tasks for each CaseID',fontsize=15)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show() 

# =============================================================================
# 사원별 업무 종류
# =============================================================================
origin = target4.groupby('originator').sum()
origin = origin.drop('caseID', axis = 1)


plt.figure(figsize=(15,10))
ax = sns.heatmap(origin,cmap='Greens',vmax=500)
plt.title('Tasks for each Originator',fontsize=15)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show() 
     
# =============================================================================
# 사원별 업무 횟수 
# =============================================================================
origin_count = target4.groupby('originator').count()
origin_count = origin_count['caseID']
origin_count = pd.DataFrame(origin_count)
origin_count.reset_index(inplace = True)
origin_count = origin_count.sort_values('caseID', ascending = False)

plt.figure(figsize=(20,10))
sns.barplot(x='caseID',y='originator', data = origin_count, orient = 'h')
plt.title('Frequency of employees\' task' ,fontsize=15)
plt.show()


# =============================================================================
# 프로세스마이닝
# =============================================================================
import pm4py

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import factory as dfg_factory
from pm4py.visualization.dfg import factory as dfg_vis_factory
from pm4py.objects.conversion.dfg import factory as dfg_mining_factory
from pm4py.visualization.process_tree import factory as pt_vis_factory
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.algo.discovery.inductive import factory as inductive_miner
from pm4py.algo.discovery.alpha import factory as alpha_miner
from pm4py.visualization.petrinet import factory as vis_factory
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.graphs import visualizer as graphs_visualizer
from pm4py.algo.discovery.heuristics import factory as heuristics_miner
from pm4py.visualization.petrinet import factory as hn_vis_factory
from pm4py.visualization.petrinet import factory as pn_vis_factory


p_target = target2[['caseID','taskID','eventtype','timestamp']]
p_target.sort_values('caseID')
p_target = p_target.drop('eventtype', axis = 1)

colname = ['case:concept:name', 'concept:name', 'time:timestamp']
p_target.columns = colname

p_target['case:concept:name'] = p_target['case:concept:name'].astype(str)

#로그변환
log = log_converter.apply(p_target, variant=log_converter.Variants.TO_DATA_FRAME) 

# =============================================================================
# DFG
# =============================================================================
from pm4py.algo.discovery.dfg import algorithm as dfg_algorithm

dfg = dfg_algorithm.apply(log)
net, im, fm = dfg_mining_factory.apply(dfg)
     
gviz = dfg_vis_factory.apply(dfg, log = log, variant='frequency')
dfg_vis_factory.view(gviz)

gviz = dfg_vis_factory.apply(dfg, log = log, variant='performance')
dfg_vis_factory.view(gviz)


# =============================================================================
# 알파알고리즘
# =============================================================================
al_net, al_ini, al_fin = alpha_miner.apply(log)

al_gviz = vis_factory.apply(al_net, al_ini, al_fin)
vis_factory.view(al_gviz)
     
# =============================================================================
# 휴리스틱 
# =============================================================================
heu_net, heu_im, heu_fm = heuristics_miner.apply(log, parameters={'dependency_thresh':0.6, 'min_act_count':2})
     

gviz_heu = pn_vis_factory.apply(heu_net, heu_im, heu_fm)
pn_vis_factory.view(gviz_heu)



# =============================================================================
# 알파 알고리즘
# =============================================================================
import os
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.algo.discovery.alpha import factory as alpha_miner
from pm4py.visualization.petrinet import factory as vis_factory

#log에 alpha algorithm을 적용
net, initial_marking, final_marking = alpha_miner.apply(log) 
#도출된 petri net을 visualization
gviz = vis_factory.apply(net, initial_marking, final_marking)
vis_factory.view(gviz)


# =============================================================================
# 
# =============================================================================

pp_target = target[['caseID','taskID','eventtype','timestamp']]
pp_target.sort_values('caseID')

colname = ['case:concept:name', 'concept:name', 'time:timestamp']
p_target.columns = colname

p_target['case:concept:name'] = p_target['case:concept:name'].astype(str)

#로그변환
log = log_converter.apply(p_target, variant=log_converter.Variants.TO_DATA_FRAME) 


parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY : 'case:concept:name'}
event_log = log_converter.apply(log, parameters = parameters, variant = log_converter.Variants.TO_EVENT_LOG)

from pm4py.objects.conversion.log.converter import to_data_frame

df = to_data_frame.apply(event_log)
df.head()
df.info()
from pm4py.algo.filtering.pandas.attributes import attributes_filter

activities = attributes_filter.get_attribute_values(df, attribute_key = "concept:name")
activities
    
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer


heu_net = heuristics_miner.apply_heu(log)

from pm4py.algo.discovery.heuristics import factory as heuristics_miner
heu_net = heuristics_miner.apply_heu(log, parameters={"dependency_thresh": 0.6, "min_act_count" : 2})

gviz = hn_visualizer.apply(heu_net)
hn_visualizer.view(gviz)


# =============================================================================
# 데이터 개요
# =============================================================================
## 파이차트
labels = ['Ex','Personal','Phone','Web'] ## 라벨
frequency = [220,226,250,250] ## 빈도
colors = ['#DFE3DB', '#DDECCA', '#2FA599', '#78B4AD']
wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}

plt.pie(frequency, labels=labels, autopct='%.1f%%', startangle=260, counterclock=False, colors=colors, wedgeprops=wedgeprops)
plt.title('1970\'s Contact Method')
plt.show()