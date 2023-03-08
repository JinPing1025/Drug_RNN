
import numpy as np
import pandas as pd
import random 


data = pd.read_csv("datasets\SARS.CSV")
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

from Measure import smi_mol

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



from sklearn import svm
svm = svm.SVC()
scores = cross_val_score(svm,data_X, data_Y, cv=5)
scores.mean()

svm.fit(X_train, y_train)
rf_disp = RocCurveDisplay.from_estimator(svm, X_test, y_test)
plt.show()


from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(random_state=1)
scores = cross_val_score(logistic,data_X, data_Y, cv=5)
scores.mean()

logistic.fit(X_train, y_train)
rf_disp = RocCurveDisplay.from_estimator(logistic, X_test, y_test)
plt.show()

from sklearn.ensemble import AdaBoostClassifier
AdaBoost = AdaBoostClassifier(n_estimators=150)
scores = cross_val_score(AdaBoost,data_X, data_Y, cv=5)
scores.mean()

AdaBoost.fit(X_train, y_train)
rf_disp = RocCurveDisplay.from_estimator(AdaBoost, X_test, y_test)
plt.show()

