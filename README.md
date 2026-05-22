# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
data
```

<img width="1247" height="760" alt="image" src="https://github.com/user-attachments/assets/46dfeff9-d0d1-42ea-b6a5-860840c9c250" />


```
data.isnull().sum()
```

<img width="277" height="310" alt="image" src="https://github.com/user-attachments/assets/be2df6cd-b88e-4f2f-bf5c-cc78383b67ac" />


```
missing=data[data.isnull().any(axis=1)]
missing
```

<img width="1255" height="751" alt="image" src="https://github.com/user-attachments/assets/5adb58d3-71f8-40e7-9851-718f9dabdb32" />


```
data2=data.dropna(axis=0)
data2
```

<img width="1248" height="770" alt="image" src="https://github.com/user-attachments/assets/f8703255-8107-4181-981e-fb03d5830aef" />


```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

<img width="462" height="263" alt="image" src="https://github.com/user-attachments/assets/da610988-862b-44ef-b99a-10af196f3c06" />


```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

<img width="358" height="445" alt="image" src="https://github.com/user-attachments/assets/84d9928f-4614-4127-b120-ad743b782186" />


```
 data2
```

<img width="1256" height="575" alt="image" src="https://github.com/user-attachments/assets/958002b7-40c7-441a-881f-1f85a2b076bd" />


```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

<img width="1127" height="458" alt="image" src="https://github.com/user-attachments/assets/fb920ac1-b035-4150-aa24-4ce26a661f75" />


```
columns_list=list(new_data.columns)
print(columns_list)
```

<img width="1110" height="380" alt="image" src="https://github.com/user-attachments/assets/a31d15b5-eb18-4983-a228-15cf4884e810" />


```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

<img width="1127" height="377" alt="image" src="https://github.com/user-attachments/assets/cef6beb2-7502-4871-bf6b-a04b10ac5944" />


```
y=new_data['SalStat'].values
print(y)
```

<img width="188" height="34" alt="image" src="https://github.com/user-attachments/assets/219ef24e-eb66-4b20-af4b-6f493113d7d4" />


```
x=new_data[features].values
print(x)
```

<img width="399" height="151" alt="image" src="https://github.com/user-attachments/assets/916f47e3-8350-4bec-bd96-9eef532bfbb9" />


```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```

<img width="303" height="87" alt="image" src="https://github.com/user-attachments/assets/a7da084e-7e41-460d-b157-7c7299e68ce5" />


```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```


<img width="185" height="48" alt="image" src="https://github.com/user-attachments/assets/f756cb7f-3247-4645-aa56-f1f7b1f47dba" />


```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```

<img width="217" height="33" alt="image" src="https://github.com/user-attachments/assets/d0df2e51-3396-497b-ab92-fa9d402b6337" />


```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```

<img width="345" height="34" alt="image" src="https://github.com/user-attachments/assets/a443b533-db20-420c-9a3e-89fe0ad6de40" />


```
data.shape
```

<img width="177" height="40" alt="image" src="https://github.com/user-attachments/assets/7ad689ec-da33-4342-a3a2-18cc8cd582af" />


```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
'Feature1': [1,2,3,4,5],
'Feature2': ['A','B','C','A','B'],
'Feature3': [0,1,1,0,1],
'Target'  : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```


<img width="349" height="51" alt="image" src="https://github.com/user-attachments/assets/5ed68830-3abe-4f74-a9d5-9646ed88f9c1" />


```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```


<img width="428" height="182" alt="image" src="https://github.com/user-attachments/assets/847043a3-52d0-4806-8894-1f5a46c701ff" />


```
tips.time.unique()
```

<img width="410" height="52" alt="image" src="https://github.com/user-attachments/assets/40236575-58b4-4f1b-8ed9-a31876002a7b" />


```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

<img width="292" height="91" alt="image" src="https://github.com/user-attachments/assets/a20ab944-d589-4dbe-8ef4-b49424d96097" />


```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```


<img width="448" height="55" alt="image" src="https://github.com/user-attachments/assets/4dd5fac0-2c37-43ed-823f-6e800c32b52b" />


# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
