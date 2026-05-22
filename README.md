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

<img width="1123" height="671" alt="image" src="https://github.com/user-attachments/assets/d7bd8d24-b3ea-4ee1-b9b7-5b1f6ecbe8d7" />


```
data.isnull().sum()
```

<img width="269" height="273" alt="image" src="https://github.com/user-attachments/assets/76835a00-ee7f-4017-8840-96699d281322" />


```
missing=data[data.isnull().any(axis=1)]
missing
```

<img width="1123" height="663" alt="image" src="https://github.com/user-attachments/assets/34ba90ed-45a9-4d6e-bee3-b02f0671dc33" />


```
data2=data.dropna(axis=0)
data2
```

<img width="1122" height="682" alt="image" src="https://github.com/user-attachments/assets/9c316886-375f-49d3-9604-8fd4a3a02c02" />


```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

<img width="520" height="238" alt="image" src="https://github.com/user-attachments/assets/d0a56142-9482-4329-a4dd-060c6188c4fa" />


```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

<img width="359" height="401" alt="image" src="https://github.com/user-attachments/assets/e3453328-aa23-45ab-aad9-53fe2a748eff" />


```
 data2
```

<img width="1126" height="522" alt="image" src="https://github.com/user-attachments/assets/017cf4d3-3123-4aef-ab88-41e08b1aaeaa" />


```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

<img width="1132" height="451" alt="image" src="https://github.com/user-attachments/assets/3a23a37e-e5f9-4a41-83de-36807eaa9f1e" />


```
columns_list=list(new_data.columns)
print(columns_list)
```

<img width="1123" height="378" alt="image" src="https://github.com/user-attachments/assets/33475a31-1618-490b-839c-2ef9e2d5b88f" />


```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

<img width="1123" height="380" alt="image" src="https://github.com/user-attachments/assets/675198c6-5d8c-47c7-be3b-65b1ce5d726b" />


```
y=new_data['SalStat'].values
print(y)
```

<img width="202" height="36" alt="image" src="https://github.com/user-attachments/assets/78a3f89b-bc79-4914-8264-38f9ff197b17" />


```
x=new_data[features].values
print(x)
```

<img width="432" height="149" alt="image" src="https://github.com/user-attachments/assets/16d1fad0-6ed4-4bdb-a9f9-61cb69884a27" />


```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```

<img width="304" height="96" alt="image" src="https://github.com/user-attachments/assets/4e84d0f9-49bd-4752-80d4-f0ba82801cb8" />


```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

<img width="227" height="59" alt="image" src="https://github.com/user-attachments/assets/cb702356-d967-4784-934f-eb69658ebcbb" />


```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```

<img width="213" height="37" alt="image" src="https://github.com/user-attachments/assets/25454c9e-2869-4509-bdcb-9168e5307236" />


```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```

<img width="288" height="29" alt="image" src="https://github.com/user-attachments/assets/1411ec8e-165e-4f6f-b187-651aad127294" />


```
data.shape
```

<img width="161" height="37" alt="image" src="https://github.com/user-attachments/assets/0fb939da-d6c1-47bc-9eab-093468a76c17" />


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

<img width="392" height="53" alt="image" src="https://github.com/user-attachments/assets/61ca8ad7-96c9-450d-ad4c-1469301042a9" />


```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

<img width="431" height="198" alt="image" src="https://github.com/user-attachments/assets/40f717ce-c751-4804-b231-0cea46f5f169" />


```
tips.time.unique()
```

<img width="409" height="59" alt="image" src="https://github.com/user-attachments/assets/4cd49a9f-7ae3-4ef5-9363-19f61694eb9a" />


```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

<img width="248" height="87" alt="image" src="https://github.com/user-attachments/assets/ea304f57-cd03-450a-b3cf-87c8f531cc34" />


```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

<img width="387" height="50" alt="image" src="https://github.com/user-attachments/assets/8474c5d3-cedf-4cd9-b5f3-231cef5f0213" />

# RESULT:
  Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
