
#导入相关库
import pandas as pd
import numpy as np

#导入数据
data=pd.read_csv(r'datasets\nwe_smi_RO5.csv')
print(data)
dataset = list(data.smiles)


from rdkit import Chem
from calculate_property import calculate_qed
from Metrics import compute_fragments
from Metrics import cos_similarity

# def smi_mol(dataset):
#     Smiles = []
#     for i in dataset:
#         mol = Chem.MolFromSmiles(i)
#         Smiles.append(mol)
#     return Smiles 
# mol_data = smi_mol(dataset)

'''QED_score'''
data_QED = calculate_qed(dataset)

'''Frag_score'''

N3 = 'Cc1cc(no1)C(=O)N[C@@H](C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C[C@@H]2CCNC2=O)C=CC(=O)OCc3ccccc3'

smi = Chem.MolFromSmiles(N3)
N3 = Chem.MolToSmiles(smi, isomericSmiles=False)
N3 = [N3]

def remove_none(dataset):
    generator_data=[]
    generator_mol=[]
    generator_none=[]
    for i in range(len(dataset)):
        mol = dataset[i]
        m = Chem.MolFromSmiles(mol)
        if m is None:
            generator_none.append(m)
            print(i)
        else:
            generator_data.append(dataset[i])
            generator_mol.append(m)
    return generator_data,generator_mol

gen_data,gen_mol = remove_none(dataset)
N3_data,N3_mol = remove_none(N3)


frag_value = []
for i in range(len(gen_mol)):
    generator_frag = compute_fragments([gen_mol[i]], n_jobs=1)
    data_frag = compute_fragments(N3_mol, n_jobs=1)
    Frag_score = cos_similarity(data_frag, generator_frag)
    print('Frag: ',Frag_score)
    frag_value.append(Frag_score)


'''Similarity to a nearest neighbor'''
from Metrics import SNN
from Metrics import fingerprint

generator_Fps = fingerprint(gen_mol,fp_type='morgan')

N3_Fps = fingerprint(N3_mol,fp_type='morgan')
N3_Fps = np.concatenate((N3_Fps,N3_Fps),axis=0)

SNN_value = []
for i in range(len(generator_Fps)):
    a = generator_Fps[i]
    b = np.vstack((a,a))
    SNN_score = SNN(np.array(b),np.array(N3_Fps),p=1)
    print('SNN:',SNN_score)
    SNN_value.append(SNN_score)
    
    
'''data_feature '''
data_frag = frag_value
data_SNN = SNN_value
feature = [list(t) for t in zip(data_SNN,data_frag)]
data_feature  = pd.DataFrame(data = feature ,columns=['SNN','Frag'])


#极小型指标 -> 极大型指标
def dataDirection_1(datas):
    return np.max(datas)-datas     

#中间型指标 -> 极大型指标
def dataDirection_2(datas, x_best):
    temp_datas = datas - x_best
    M = np.max(abs(temp_datas))
    answer_datas = 1 - abs(datas - x_best) / M     
    return answer_datas
    
#区间型指标 -> 极大型指标
def dataDirection_3(datas, x_min, x_max):
    M = max(x_min - np.min(datas), np.max(datas) - x_max)
    answer_list = []
    for i in datas:
        if(i < x_min):
            answer_list.append(1 - (x_min-i) /M)      
        elif( x_min <= i <= x_max):
            answer_list.append(1)
        else:
            answer_list.append(1 - (i - x_max)/M)
    return np.array(answer_list)   


def Standard(datas):
    K_list = []
    K = np.power(np.sum(pow(datas,2),axis = 0),0.5)
    for i in range(len(K)):
        a = datas[: , i] / K[i]
        K_list.append(a)
    S_list = np.array(K_list).T
    return S_list 

sta_data = Standard(data_feature.values)
sta_data

def Score(sta_data):
    z_max = np.amax(sta_data , axis=0)
    z_min = np.amin(sta_data , axis=0)
    # 计算每一个样本点与最大值的距离
    tmpmaxdist = np.power(np.sum(np.power((z_max - sta_data),2),axis = 1),0.5)  # 每个样本与Z+的距离
    tmpmindist = np.power(np.sum(np.power((z_min - sta_data),2),axis = 1),0.5)  # 每个样本与Z+的距离
    score = tmpmindist / (tmpmindist + tmpmaxdist)
    score = score / np.sum(score)  # 归一化处理
    return score

sco = Score(data_feature)*1000
sco

data_feature['score'] = sco
data_feature.insert(0,'smiles',dataset)
data_sort = data_feature.sort_values(by='score',ascending=False)

smi_sort = data_sort
smi_sort.to_csv('datasets/new_smi_sort.csv')



# #熵权法计算权重
# def entropyWeight(data):
#     P = np.array(data)
#     # 计算熵值
#     E = np.nansum(-P * np.log(P) / np.log(len(data)), axis=0)
#     # 计算权系数
#     return (1 - E) / (1 - E).sum()

# weight = entropyWeight(sta_data)

