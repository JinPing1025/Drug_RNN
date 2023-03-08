###读取生成数据
import numpy as np
import pandas as pd
import random 

gen_data = pd.read_csv("generate\TF=20.csv")
gen_smi = list(gen_data.smiles)

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

def smi_mol(gen_smi):
    mols = []
    for i in gen_smi:
        mol = Chem.MolFromSmiles(i)
        mols.append(mol)
    return mols

gen_mol = smi_mol(gen_smi)

a = fingerprint(gen_mol,fp_type='morgan')
gen_dataset = np.array(a,dtype='int64')


#预训练机器学习数据集
import numpy as np
import pandas as pd
import random 

data = pd.read_csv("datasets/SARS.CSV")
smi = list(data.Smiles)
activity = np.array(data.activity,dtype='int32')


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


train_mol = smi_mol(smi)
a = fingerprint(train_mol,fp_type='morgan')

a = pd.DataFrame(data=a)
b = pd.DataFrame(data=activity)
dfs = [a,b]
result = pd.concat(dfs,axis=1)

dataset = np.array(result,dtype='int64')

np.random.seed(5)
np.random.shuffle(dataset)

data_X = dataset[:,:1024]        
data_Y= dataset[:,1024]   

from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.3, random_state=42)



from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=130, max_depth=None,
    min_samples_split=2, random_state=0)
scores = cross_val_score(RF,data_X, data_Y, cv=5)
scores.mean()

RF.fit(X_train, y_train)
rf_disp = RocCurveDisplay.from_estimator(RF, X_test, y_test)
plt.show()

#分类生成数据集得到活性
RF_activity = RF.predict(gen_dataset)

a = pd.DataFrame(data=gen_smi,columns=['smiles'])
b = pd.DataFrame(data=RF_activity, columns=['activity'])
dfs = [a,b]
RF_result = pd.concat(dfs,axis=1)
RF_result.to_csv('datasets/RF_result.csv')


from sklearn import svm
svm = svm.SVC()
scores = cross_val_score(svm,data_X, data_Y, cv=5)
scores.mean()

svm.fit(X_train, y_train)
svm_disp = RocCurveDisplay.from_estimator(svm, X_test, y_test)
plt.show()

#分类生成数据集得到活性
svm_activity = svm.predict(gen_dataset)

a = pd.DataFrame(data=gen_smi,columns=['smiles'])
b = pd.DataFrame(data=svm_activity, columns=['activity'])
dfs = [a,b]
svm_result = pd.concat(dfs,axis=1)
svm_result.to_csv('datasets/svm_result.csv')


from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(random_state=1)
scores = cross_val_score(logistic,data_X, data_Y, cv=5)
scores.mean()

logistic.fit(X_train, y_train)
lg_disp = RocCurveDisplay.from_estimator(logistic, X_test, y_test)
plt.show()

#分类生成数据集得到活性
logistic_activity = logistic.predict(gen_dataset)

a = pd.DataFrame(data=gen_smi,columns=['smiles'])
b = pd.DataFrame(data=logistic_activity, columns=['activity'])
dfs = [a,b]
logistic_result = pd.concat(dfs,axis=1)
logistic_result.to_csv('datasets/logistic_result.csv')


from sklearn.ensemble import AdaBoostClassifier
AdaBoost = AdaBoostClassifier(n_estimators=150)
scores = cross_val_score(AdaBoost,data_X, data_Y, cv=5)
scores.mean()

AdaBoost.fit(X_train, y_train)
AdaBoost_disp = RocCurveDisplay.from_estimator(AdaBoost, X_test, y_test)
plt.show()


#分类生成数据集得到活性
AdaBoost_activity = AdaBoost.predict(gen_dataset)

a = pd.DataFrame(data=gen_smi,columns=['smiles'])
b = pd.DataFrame(data=AdaBoost_activity, columns=['activity'])
dfs = [a,b]
AdaBoost_result = pd.concat(dfs,axis=1)
AdaBoost_result.to_csv('datasets/AdaBoost_result.csv')



def activity_num(results):
    a = 0
    for i in range(len(gen_data.smiles)):
        if results.activity[i]==1:
            a +=1
    return a


print('RF_activity:',activity_num(RF_result))
print('svm_activity:',activity_num(svm_result))
print('logistic_activity:',activity_num(logistic_result))
print('AdaBoost_activity:',activity_num(AdaBoost_result))


overall_activity = []

for i in range(len(gen_data.smiles)):
    if RF_result.activity[i]==1 and svm_result.activity[i]==1 and \
        logistic_result.activity[i]==1 and AdaBoost_result.activity[i]==1:
            overall_activity.append(gen_data.smiles[i])

smi_activity = pd.DataFrame(data=overall_activity,columns=['smiles'],index=None)
smi_activity.to_csv('datasets/smi_activity11.csv')


# plt.figure(dpi=300,figsize=(6,4))
# ax = plt.gca()
# rf_disp = RocCurveDisplay.from_estimator(RF, X_test, y_test, ax=ax, alpha=1.0)
# svm_disp = RocCurveDisplay.from_estimator(svm, X_test, y_test, ax=ax, alpha=1.0)
# lg_disp = RocCurveDisplay.from_estimator(logistic, X_test, y_test, ax=ax, alpha=1.0)
# AdaBoost_disp = RocCurveDisplay.from_estimator(AdaBoost, X_test, y_test, ax=ax, alpha=1.0)

# plt.show()




