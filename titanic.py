import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#將資料集切成訓練集和驗證集的工具
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#模型評估指標函數
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#讀取檔案
train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')


#檢查缺失值
#.isnull():檢查是否空格
print("=== 補值前 train_df 缺失值 ===")
print(train_df.isnull().sum())
print("\n=== 補值前 test_df 缺失值 ===")
print(test_df.isnull().sum())

#補缺漏值，使用中位數、平均數、眾數
#.fillna()：補缺失值
# inplace=True:修改原有資料
train_df.fillna({'Age':train_df['Age'].median(),
                'Embarked':train_df['Embarked'].mode()[0],'Cabin':'U'},inplace=True)#缺失的艙位用'U'表示未知
test_df.fillna({'Age': test_df['Age'].median(),
                'Fare': test_df['Fare'].mean(),'Cabin':'U'},inplace=True)


#再次檢查是否還有缺失值
print("=== train_df 缺值 ===")
print(train_df.isnull().values.any())

print("=== test_df 缺值 ===")
print(test_df.isnull().sum())

#生還者與死亡分佈
survived_counts = train_df['Survived'].value_counts()

#繪製圓餅圖
survived_counts.plot.pie(autopct='%1.1f%%', startangle=90, labels=["Dead", "Survived"], colors=sns.color_palette("Blues", 2))
plt.title('Survival Distribution')
plt.ylabel('')  # 去掉 y 軸標籤
plt.show()

#計算性別和生存率
#.groupby()：將數據進行分組
#.reset_index():恢復索引，將sex列恢復普通列
sex_survival=train_df.groupby('Sex')['Survived'].mean().reset_index()

#指定顏色
colors=['lightblue' if sex == 'male' else 'lightpink' for sex in sex_survival['Sex']]

#使用seaborn繪製長條圖
sns.barplot(x='Sex', y='Survived', data=sex_survival, palette=colors)
plt.title('Survival Rate by Sex', fontsize=14, fontweight='bold')#fontweight:字體粗體
plt.xlabel('Sex', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.ylim(0, 1)  # y軸範圍：0-1
plt.tight_layout() #自動調整圖表的版面配置，避免重疊
plt.show()

#年齡分組
train_df['AgeGroup'] = pd.cut(train_df['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80], 
                           labels=['0-10', '11-21','21-30', '31-40', '41-50', '51-60', '61-70','71-80'], right=False)

#計算年齡和生存率
age_survival=train_df.groupby('AgeGroup')['Survived'].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(x='AgeGroup', y='Survived', data=age_survival, palette='Blues')
plt.title('Survival Rate by Age Group')
plt.ylabel('Survival Rate')
plt.xlabel('Age Group')
plt.ylim(0, 1) 
plt.tight_layout() 
plt.show()

#計算艙等和生存率
pclass_survival=train_df.groupby('Pclass')['Survived'].mean().reset_index()

plt.figure(figsize=(6, 4))
sns.barplot(x='Pclass', y='Survived', data=pclass_survival, palette='pastel')
plt.title('Survival Rate by PClass')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

#建立票價分組
train_df['FareGroup'] = pd.cut(train_df['Fare'], bins=[0, 10, 30, 100, 600], 
                               labels=['0-10', '10-30', '30-100', '100+'])

#計算票價和生存率
fare_survival=train_df.groupby('FareGroup')['Survived'].mean().reset_index()

sns.barplot (x='FareGroup', y='Survived', data=fare_survival, palette ='Greens')
plt.title('Survival Rate by Fare Group')
plt.ylabel('Fare')
plt.ylim(0, 1) 
plt.tight_layout() 
plt.show()

#特徵工程
# 從姓名中提取稱謂 (Mr, Miss, etc.)
for dataset in [train_df, test_df]:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# 將罕見的稱謂合併成一類
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 
                                                  'Don', 'Dr', 'Major', 'Rev', 'Sir', 
                                                  'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
# 將 Title 轉換為數值類別
    dataset['Title']=dataset['Title'].map ({'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5})
    dataset['Title'] = dataset['Title'].fillna(0)  # 防止有缺漏

#IsAlone 變數
for dataset in [train_df, test_df]:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 0, 'IsAlone'] = 1

# Fare分組
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4, labels=[0, 1, 2, 3])
fare_bins=pd.qcut(train_df['Fare'],4,retbins=True)[1]
test_df['FareBand'] = pd.cut(test_df['Fare'], bins=fare_bins, labels=False,include_lowest=True)

# 將 FareBand 當作新變數使用
train_df['FareBand'] = train_df['FareBand'].astype(int)
test_df['FareBand'] = test_df['FareBand'].astype(int)

#建立訓練集資料與標籤
#資料前處理(挑選較有關的特徵)
#features:建立新清單
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked','Title','IsAlone','FareBand']
X = train_df[features]
y = train_df['Survived']


#處理類別變數（將文字轉換為數字）
#pd.get_dummies():將類別變數轉換成0或1
#drop_first=True：避免多重共線性問題
X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)


#切分資料與訓練模型
#sklearn:切分訓練與測試資料（20%的驗證，80%的訓練）
#train_test_split：隨機將資料集分成兩部分
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#訓練模型
#LogisticRegression:邏輯回歸模型
#max_iter=1000：讓模型跑1000次
model = LogisticRegression(max_iter=1000)  # max_iter 提高是為了避免收斂問題
model.fit(X_train, y_train)

#模型評估
y_pred = model.predict(X_val)
print("準確率 (Accuracy):", accuracy_score(y_val, y_pred))
print("混淆矩陣:")
print(confusion_matrix(y_val, y_pred))
print("分類報告:")
print(classification_report(y_val, y_pred))

#測試集特徵處理（與訓練集相同）
test_X = test_df[features]
test_X = pd.get_dummies(test_X, columns=['Sex', 'Embarked'], drop_first=True)

# 缺失值補齊（用訓練集中位數）
test_X['Age'].fillna(X['Age'].median(), inplace=True)

# 補齊缺少的欄位，確保 test_df欄位與 X 完全一致
for col in X.columns:
    if col not in test_X.columns:
        test_X[col] = 0

# 確保欄位順序一致
test_X = test_X[X.columns]

# 用模型預測測試資料
test_predictions = model.predict(test_X)

# 產出提交檔與儲存結果
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_predictions
})
submission.to_csv('titanic_submission.csv', index=False)








