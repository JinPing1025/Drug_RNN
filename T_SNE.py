###读取生成数据
import numpy as np
import pandas as pd
import random 

# gen_data = pd.read_csv("datasets\smi_activity.csv")
# gen_smi = list(gen_data.smiles)

gen_data = pd.read_csv("datasets\smi_activity11.csv")
gen_smi = list(gen_data.smiles)

act_data = pd.read_csv("datasets\SARS.csv")
act_smi = list(act_data.Smiles[:622])


from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

def fingerprint(dataset,fp_type='maccs'):
    if fp_type == 'maccs':
        Fps = [MACCSkeys.GenMACCSKeys(x) for x in dataset]
    elif fp_type == 'morgan':
        Fps =[AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=1024,useFeatures=True) for x in dataset] 
        Fps = np.asarray(Fps)
    return Fps

gen_mols = []
for i in gen_smi:
    mol = Chem.MolFromSmiles(i)
    gen_mols.append(mol)

act_mols = []
for i in act_smi:
    mol = Chem.MolFromSmiles(i)
    act_mols.append(mol)


a = fingerprint(gen_mols,fp_type='morgan')
gen_dataset = np.array(a,dtype='int64')

b = fingerprint(act_mols,fp_type='morgan')
act_dataset = np.array(b,dtype='int64')


a1 = np.array([np.insert(i, 1024, np.array([1])) for i in a])
b1 = np.array([np.insert(i, 1024, np.array([0])) for i in b])

c1 = np.concatenate((a1,b1)) 

tsne_1 = TSNE(n_components=2, learning_rate=500).fit_transform(c1[:,:-1])
plt.figure(figsize=(12, 6),dpi=800)
plt.scatter(tsne_1[:, 0], tsne_1[:, 1], c=c1[:,-1],s=4)##按Y值区分散点

plt.colorbar()
plt.show()


'''Frag,Scaf'''
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

gen_data,gen_mol = remove_none(gen_smi)
act_data,act_mol = remove_none(act_smi)


'''Fragment similarity'''
from Metrics import compute_fragments
from Metrics import cos_similarity

generator_frag = compute_fragments(gen_mol, n_jobs=1)
data_frag = compute_fragments(act_mol, n_jobs=1)

Frag_score = cos_similarity(data_frag, generator_frag)
print('Frag: ',Frag_score)


'''Scaffold similarity (Scaff)'''
from Metrics import compute_scaffolds

generator_scaf = compute_scaffolds(gen_mol, n_jobs=1, min_rings=2)
data_scaf = compute_scaffolds(act_mol, n_jobs=1, min_rings=2)

Scaf_score = cos_similarity(data_scaf, generator_scaf)
print('Scaf: ',Scaf_score)


# 选择维度
tsne = TSNE(n_components=3, random_state=0)
tsne_obj= tsne.fit_transform(c1[:,:-1])
tsne_df = pd.DataFrame({'X':tsne_obj[:,0],'Y':tsne_obj[:,1],'Z':tsne_obj[:,2],'digit':c1[:,-1]})

plt.figure(figsize=(12,8), dpi= 100) #创建画布并设定画布大小
sns.scatterplot(tsne_1[:, 0], tsne_1[:, 1],hue="digit",palette=['red','blue'],legend='full',data=tsne_df)
plt.legend(fontsize=16)
plt.show()
