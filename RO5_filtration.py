
###读取生成数据
import numpy as np
import pandas as pd
import random 

gen_data = pd.read_csv('datasets/novel_mol.csv')
gen_smi = list(gen_data.smiles)

from calculate_property import density_estimation
from calculate_property import calculate_MW, calculate_LogP
from calculate_property import calculate_NumHDonors
from calculate_property import calculate_NumHAcceptors
from calculate_property import calculate_NumRotatableBonds
import matplotlib.pyplot as plt

#######################################筛选前#######################################
'''Molecular weight (MW)'''
generator_MW = calculate_MW(gen_smi)
num = 100
generator_MWs = [i/num for i in generator_MW] 

'''LogP'''
generator_LogP = calculate_LogP(gen_smi)

'''NumHDonors'''
generator_NumHDonors = calculate_NumHDonors(gen_smi)

'''NumHAcceptors'''
generator_NumHAcceptors = calculate_NumHAcceptors(gen_smi)

'''NumRotatableBonds'''
generator_NumRotatableBonds = calculate_NumRotatableBonds(gen_smi)


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8), dpi= 100) #创建画布并设定画布大小
sns.set(color_codes=True)
sns.set_style("white")

ax1 = sns.kdeplot(generator_MWs,color="b",shade=True,label="generator_mol_MW")
ax2 = sns.kdeplot(generator_LogP,color="r",shade=True,label="generator_mol_logP")
ax3 = sns.kdeplot(generator_NumHDonors ,color="g",shade=True,label="generator_mol_HBD")
ax4 = sns.kdeplot(generator_NumHAcceptors,color="y",shade=True,label="generator_mol_HBA")
ax5 = sns.kdeplot(generator_NumRotatableBonds ,color="m",shade=True,label="generator_mol_RotB")

plt.legend(fontsize=16)  
plt.show()   

#######################################筛选后#######################################
"""
five rule of Lipinski  
MW <= 500, LogP <= 5, NumHDonors <= 5, NumHAcceptors <= 10, NumRotatableBonds <= 10
"""

a = pd.DataFrame(data=gen_smi,columns=['smiles'])
b = pd.DataFrame(data=generator_MW, columns=['values'])
dfs = [a,b]
MW_result = pd.concat(dfs,axis=1)


a = pd.DataFrame(data=gen_smi,columns=['smiles'])
b = pd.DataFrame(data=generator_LogP, columns=['values'])
dfs = [a,b]
LogP_result = pd.concat(dfs,axis=1)


a = pd.DataFrame(data=gen_smi,columns=['smiles'])
b = pd.DataFrame(data=generator_NumHDonors, columns=['values'])
dfs = [a,b]
NumHDonors_result = pd.concat(dfs,axis=1)


a = pd.DataFrame(data=gen_smi,columns=['smiles'])
b = pd.DataFrame(data=generator_NumHAcceptors, columns=['values'])
dfs = [a,b]
NumHAcceptors_result = pd.concat(dfs,axis=1)


a = pd.DataFrame(data=gen_smi,columns=['smiles'])
b = pd.DataFrame(data=generator_NumRotatableBonds, columns=['values'])
dfs = [a,b]
NumRotatableBonds_result = pd.concat(dfs,axis=1)


overall_RO5 = []
for i in range(len(gen_data.smiles)):
    if LogP_result.values[i][1] <= 5 and NumHDonors_result.values[i][1] <= 5 and\
        NumHAcceptors_result.values[i][1] <= 10 :
            overall_RO5.append(gen_data.smiles[i])

smi_RO5 = pd.DataFrame(data=overall_RO5,columns=['smiles'],index=None)
smi_RO5.to_csv('datasets/nwe_smi_RO5.csv')


'''Molecular weight (MW)'''
generator_MW = calculate_MW(overall_RO5)
num = 100
generator_MWs = [i/num for i in generator_MW] 

'''LogP'''
generator_LogP = calculate_LogP(overall_RO5)

'''NumHDonors'''
generator_NumHDonors = calculate_NumHDonors(overall_RO5)

'''NumHAcceptors'''
generator_NumHAcceptors = calculate_NumHAcceptors(overall_RO5)

'''NumRotatableBonds'''
generator_NumRotatableBonds = calculate_NumRotatableBonds(overall_RO5)


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8), dpi= 100) #创建画布并设定画布大小
sns.set(color_codes=True)
sns.set_style("white")

ax1 = sns.kdeplot(generator_MWs,color="b",shade=True,label="generator_mol_MW")
ax2 = sns.kdeplot(generator_LogP,color="r",shade=True,label="generator_mol_logP")
ax3 = sns.kdeplot(generator_NumHDonors ,color="g",shade=True,label="generator_mol_HBD")
ax4 = sns.kdeplot(generator_NumHAcceptors,color="y",shade=True,label="generator_mol_HBA")
ax5 = sns.kdeplot(generator_NumRotatableBonds ,color="m",shade=True,label="generator_mol_RotB")

plt.legend(fontsize=16)  
plt.show()   








def activity_num(results):
    a = 0
    for i in range(len(gen_data.smiles)):
        if results.values[i][1] <= 500:
            a +=1
    return a
print('MW_result:',activity_num(MW_result))

def activity_num(results):
    a = 0
    for i in range(len(gen_data.smiles)):
        if results.values[i][1] <= 5:
            a +=1
    return a
print('LogP_result:',activity_num(LogP_result))

def activity_num(results):
    a = 0
    for i in range(len(gen_data.smiles)):
        if results.values[i][1] <= 5:
            a +=1
    return a
print('NumHDonors_result:',activity_num(NumHDonors_result))

def activity_num(results):
    a = 0
    for i in range(len(gen_data.smiles)):
        if results.values[i][1] <= 10:
            a +=1
    return a
print('NumHAcceptors_activity:',activity_num(NumHAcceptors_result))

def activity_num(results):
    a = 0
    for i in range(len(gen_data.smiles)):
        if results.values[i][1] <= 10:
            a +=1
    return a
print('NumRotatableBonds_activity:',activity_num(NumRotatableBonds_result))
















