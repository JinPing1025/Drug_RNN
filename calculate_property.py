
'''QED'''
from rdkit import Chem
from rdkit.Chem.QED import qed

def calculate_qed(dataset):
    dataset_qed=[]
    for i in dataset:
        m = Chem.MolFromSmiles(i)
        a = qed(m)
        dataset_qed.append(a)
    return dataset_qed

'''SA_score'''
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw, PandasTools
from rdkit.Chem.Draw import IPythonConsole
import sascorer
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def calculate_SA(dataset):
    generator_SA = pd.DataFrame(dataset,columns=['smiles'])
    PandasTools.AddMoleculeColumnToFrame(frame=generator_SA, smilesCol='smiles')
    generator_SA['calc_SA_score'] = generator_SA.ROMol.map(sascorer.calculateScore)
    generator_SA_score = generator_SA.calc_SA_score
    return generator_SA,generator_SA_score

def draw_SA_molecule(data_SA_score,data_SA):
    (id_max, id_min) = (data_SA_score.idxmax(), data_SA_score.idxmin())
    sa_mols = [data_SA.ROMol[id_max],data_SA.ROMol[id_min]]
    img = Draw.MolsToGridImage(sa_mols, subImgSize=(340,200),
                         legends=['SA-score: {:.2f}'.format(data_SA.calc_SA_score[i]) for i in [id_max, id_min]])
    return img


from rdkit.Chem  import Descriptors

'''Molecular weight (MW)'''
def calculate_MW(dataset):
    dataset_MW=[]
    for i in dataset:
        m = Chem.MolFromSmiles(i)
        a = Descriptors.MolWt(m)
        dataset_MW.append(a)
    return dataset_MW

'''LogP'''
def calculate_LogP(dataset):
    dataset_LogP=[]
    for i in dataset:
        m = Chem.MolFromSmiles(i)
        a = Descriptors.MolLogP(m)
        dataset_LogP.append(a)
    return dataset_LogP

''' five rule of Lipinski  
    MW < 500, LogP < 5, NumHDonors < 5, NumHAcceptors < 10, NumRotatableBonds <= 10'''

from rdkit import Chem
from rdkit.Chem import Lipinski

'''NumHDonors'''
def calculate_NumHDonors(dataset):
    dataset_NumHDonors=[]
    for i in dataset:
        m = Chem.MolFromSmiles(i)
        a = Lipinski.NumHDonors(m)
        dataset_NumHDonors.append(a)
    return dataset_NumHDonors

'''NumHAcceptors'''
def calculate_NumHAcceptors(dataset):
    dataset_NumHAcceptors=[]
    for i in dataset:
        m = Chem.MolFromSmiles(i)
        a = Lipinski.NumHAcceptors(m)
        dataset_NumHAcceptors.append(a)
    return dataset_NumHAcceptors

'''NumRotatableBonds'''
def calculate_NumRotatableBonds(dataset):
    dataset_NumRotatableBonds=[]
    for i in dataset:
        m = Chem.MolFromSmiles(i)
        a = Lipinski.NumRotatableBonds(m)
        dataset_NumRotatableBonds.append(a)
    return dataset_NumRotatableBonds


'''核密度函数分布'''
import seaborn as sns

def density_estimation(password,generator_data,train_data):
    sns.set(color_codes=True)
    sns.set_style("white")
    if password == 'qed_value':
        ax1 = sns.kdeplot(generator_data,color="b",shade=True,label="generator_QED_score")
        ax2 = sns.kdeplot(train_data,color="r",shade=True,label="train_QED_score")
    elif password == 'SA_value':
        ax1 = sns.kdeplot(generator_data,color="b",shade=True,label="generator_SA_score")
        ax2 = sns.kdeplot(train_data,color="r",shade=True,label="train_SA_score")
    elif password == 'MW_value':
        ax1 = sns.kdeplot(generator_data,color="b",shade=True,label="generator_MW")
        ax2 = sns.kdeplot(train_data,color="r",shade=True,label="train_MW")
    elif password == 'LogP_value':
        ax1 = sns.kdeplot(generator_data,color="b",shade=True,label="generator_logP ")
        ax2 = sns.kdeplot(train_data,color="r",shade=True,label="train_logP")
    elif password == 'NumHDonors_value':
        ax1 = sns.kdeplot(generator_data ,color="b",shade=True,label="generator_NumHDonors ")
        ax2 = sns.kdeplot(train_data,color="r",shade=True,label="train_NumHDonors")
    elif password == 'NumHAcceptors_value':
        ax1 = sns.kdeplot(generator_data ,color="b",shade=True,label="generator_NumHAcceptors ")
        ax2 = sns.kdeplot(train_data,color="r",shade=True,label="train_NumHAcceptors")
    elif password == 'NumRotatableBonds_value':
        ax1 = sns.kdeplot(generator_data ,color="b",shade=True,label="generato_NumRotatableBonds ")
        ax2 = sns.kdeplot(train_data,color="r",shade=True,label="train_NumRotatableBonds")       
    return ax1,ax2
    