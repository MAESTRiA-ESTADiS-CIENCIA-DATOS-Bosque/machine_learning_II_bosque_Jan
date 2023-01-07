from pathlib import Path
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm

#Cargar datos usando pandas
df = pd.read_csv('loan_data_set.csv')
df.tail(10)
# shape de dataframe
df.shape
df['Credit_History'].shape
df[['Credit_History']].shape
#propiedades
df.info()
df.describe()
df.describe(exclude=np.number)
### missing info y solving
df.isnull().sum()
#‘bfill’, ‘pad’, ‘ffill’, None, sin method: 0
df.fillna(method='bfill',inplace=True)
#formato de columnas
df.isnull().sum()
df['Credit_History'].unique()
df.info()
df['Credit_History']=df['Credit_History'].astype('category')
df['ApplicantIncome']=df['ApplicantIncome'].astype('int32')
# graficos boxplot
df['LoanAmount'].plot(kind='box')
df['sqrtCoapplicantIncome'] = np.sqrt(df['CoapplicantIncome'])
df.head()
# grafico de distribuciones de la variable
plt.plot(figsize=(15,5))
sns.displot(df['CoapplicantIncome'], label='CoapplicantIncome')
sns.displot(df['sqrtCoapplicantIncome'],label='sqrtCoapplicationIncome')
df.CoapplicantIncome.skew()
df.sqrtCoapplicantIncome.skew()

#Normalizacion

## Normalizacion via Z-score( (x- mean)/std)
mean_loan=df['LoanAmount'].mean()
std_load=df['LoanAmount'].std()

df['zscoreloanamount']=(df['LoanAmount']-mean_loan)/std_load
df.head()
from sklearn.preprocessing import StandardScaler

SS=StandardScaler()
#### con scikitlearn=>  X son matrices [[X]]; y son columnas [y]
scale_loan=SS.fit_transform(df[['LoanAmount']])

### Normalizacion via (x -min)/(max -min)
min_loan=df['LoanAmount'].min()
max_loan=df['LoanAmount'].max()
df['minmaxLoanAmount']=(df['LoanAmount']-min_loan)/(max_loan-min_loan)

from sklearn.preprocessing import MinMaxScaler

MS=MinMaxScaler()

minmaxloan=MS.fit_transform(df[['LoanAmount']])

## one hot encoder

df['Property_Area'].unique()
# # urban=0  ; rural=1; semiurban=2
# # urban_ind=0  ; rural_ind=1; semiurban_ind=2
# urban              1 0 0    
# urban              1 0 0
# rural              0 1 0
# semiurban          0 0 1
# rural              0 1 0
        
# # urban_ind=0  ; rural_ind=1
# urban              1 0     
# urban              1 0 
# rural              0 1 
# semiurban          0 0 
# rural              0 1 

Property_Area_1hot = pd.get_dummies(df['Property_Area'], drop_first=True)
 
##scikitlearn

from sklearn.preprocessing import OneHotEncoder

onehot=OneHotEncoder(drop='first')

Onehot_PA= onehot.fit(df[['Property_Area','Gender']])

Onehot_PA.categories_
Onehot_PA.drop_idx_
Onehot_PA.n_features_in_
Onehot_PA.feature_names_in_

#label encoding
from sklearn.preprocessing import LabelEncoder

LE=LabelEncoder()

le_propertyA= LE.fit(df['Property_Area'])
le_propertyA.classes_
le_propertyA.transform(['Rural','Rural', 'Semiurban','Rural','Rural', 'Urban'])
le_propertyA.inverse_transform([0,0,1,0,0,2])

#tranformaciones features pd

df['LoanAmountcross']= df['LoanAmount']*df['Loan_Amount_Term']

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

def extract_year(x):
    y= x['date']
    year=y.year
    return year
df['year']= df.apply(lambda x : extract_year(x),axis=1)
## split de datos
from sklearn.model_selection import train_test_split
df_test=df[["LoanAmount","ApplicantIncome","CoapplicantIncome"]]
df_test.head()

Y=df_test['LoanAmount']
X=df_test.drop("LoanAmount", axis=1)
X.head()
Y.head()

X_train_h,X_test, Y_train_h, Y_test = train_test_split(X,Y,train_size=0.8,random_state=123)
X_train,X_val, Y_train, Y_val = train_test_split(X_train_h,Y_train_h,train_size=0.9,random_state=123)

##histogramas

sns.histplot(X_train.CoapplicantIncome,kde=True)

g=sns.FacetGrid(df, col="Property_Area", row="Gender")
g.map(sns.histplot,"LoanAmount")
plt.show()

sns.jointplot(data=df, x="ApplicantIncome", y="LoanAmount" , hue='Gender', color='b')
plt.show()

sns.jointplot(data=df, x="ApplicantIncome", y="LoanAmount" , kind='scatter', color='b')
plt.show()

sns.lmplot(x="ApplicantIncome", y="LoanAmount",hue='Property_Area',data=df)
plt.show()

corr_mat=np.corrcoef(df_test,rowvar=False)
corr_mat.shape
df_test.head()
corr_df=pd.DataFrame(corr_mat,columns=df_test.columns,index=df_test.columns)
sns.heatmap(corr_df,linewidths=1,cmap='plasma', fmt=".2f")

sns.pairplot(data=df_test,corner=True)

g= sns.PairGrid(df_test, corner=True)
g.map_lower(sns.kdeplot,hue=None, levels=5)
g.map_lower(sns.scatterplot,marker="+")
g.map_diag(sns.histplot, linewidth=0.1,kde=True)

##Regresion 
df = pd.read_csv("auto-mpg.xls")
df.head()
fig, (ax1, ax2, ax3)= plt.subplots(1,3)
df.plot.scatter(x='displacement',y='mpg',ax=ax1)
df.plot.scatter(x='horsepower',y='mpg',ax=ax2)
df.plot.scatter(x='weight',y='mpg',ax=ax3)

from sklearn.linear_model import LinearRegression

# E(Y|X) = b0+b1*X
LR = LinearRegression(fit_intercept=True)

X=df[['weight']]
Y= df['mpg']

LR.fit(X,Y)
coeft=LR.coef_
intt=LR.intercept_
LR.rank_
LR.predict([[5600]])

df_muestra= df.sample(50)
LR.fit(df_muestra[['weight']],df_muestra["mpg"])
print(LR.coef_,LR.intercept_)


for i in range(100):
    df_muestra= df.sample(50)
    LR.fit(df_muestra[['weight']],df_muestra["mpg"])
    predicciones=LR.predict(df[['weight']])
    plt.plot(df['weight'],predicciones, color='blue',alpha=0.1)

fit_av=intt+coeft*df["weight"]
plt.plot(df['weight'], fit_av, color='red')

sns.displot(df['weight'])
df['mpg'].plot(kind='box')



Xsm= sm.add_constant(df_muestra['weight'])
Y=df_muestra['mpg']

modelsm=  sm.OLS(Y,Xsm)
resultados=modelsm.fit()
resultados.summary()
b_inter,b_coef= resultados.params
resultados.bse
resultados.rsquared
resultados.pvalues

f, ax= plt.subplots()
ax.plot(df_muestra['weight'],df_muestra['mpg'],'o')
res_regresion= b_coef*df_muestra['weight']+b_inter
ax.plot(df_muestra['weight'],res_regresion, '-' )

predictor= resultados.get_prediction()
conf_bajo,conf_alto= predictor.conf_int(alpha=0.05,obs=True).T
pred_bajo,pred_alto= predictor.conf_int(alpha=0.05,obs=False).T

f, ax= plt.subplots()
ax.plot(df_muestra['weight'],df_muestra['mpg'],'o')
res_regresion= b_coef*df_muestra['weight']+b_inter
ax.plot(df_muestra['weight'],res_regresion, '-' )
ax.plot(df_muestra['weight'],conf_bajo,'--', color='red',label='cof bajo 95%', linewidth=0.2 )
ax.plot(df_muestra['weight'],conf_alto,'--', color='red',label='cof alto 95%', linewidth=0.2 )
ax.plot(df_muestra['weight'],pred_bajo,':', color='blue',label='pred bajo 95%', linewidth=0.2 )
ax.plot(df_muestra['weight'],pred_alto,':', color='blue',label='pred alta 95%', linewidth=0.2 )
ax.legend()


#Regresion lineal multiple 
# # E(Y|X1,X2,X3...) = b0+b1*X1+b2*X2+...+ bn*Xn
df.head()
X=df[['Loan_Amount_Term',"CoapplicantIncome"]]
y=df['LoanAmount']

### E( LoanAmount|LAT,CI)=>     LA  = b0+b1*LAT+b2*CI
X_train, X_test, y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=123)

from sklearn.linear_model import LinearRegression

model_mult= LinearRegression(fit_intercept=True)

model_mult.fit(X_train,y_train)
model_mult.coef_
model_mult.intercept_
prediccion= model_mult.predict(X_test)

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
r2_score(y_test,prediccion)
mean_squared_error(y_test,prediccion)
mean_absolute_error(y_test,prediccion)

df['Property_Area'].unique()

# rural      1  0
# semiurban  0  1
                 # urban      0  0
#LA = b0+b1*LAT+ b2*rural+b3*semiurban
# 
# rural=0; semiurban=1    ;b5=b0+b3
# LA = b0+b1*LAT+ b3= (b0+b3)+b1*LAT= b5+b1*LAT 
#rural=1; semiurban=0  ;b6=(b0+b2)
# LA = b0+b1*LAT+ b2= (b0+b2)+b1*LAT= b6+b1*LAT 
#rural=0; semiurban=0 
#LA = b0+b1*LAT

X=df[["Loan_Amount_Term","Property_Area"]]
y=df['LoanAmount']
X_train, X_test, y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=123)

unique= list(X_train['Property_Area'].unique())

from sklearn.preprocessing import OneHotEncoder

OHE=OneHotEncoder(categories=[unique],drop='first')
OHE.fit(X_train[['Property_Area']])

matrix=OHE.transform(X_train[['Property_Area']]).toarray()
df_oH= pd.DataFrame(matrix, columns=unique[1:],index=X_train.index)
X_train=X_train.join(df_oH)

X_train_model=X_train[["Loan_Amount_Term", "Semiurban", "Rural"]]

model_mult_cat=LinearRegression(fit_intercept=True)

model_mult_cat.fit(X_train_model, y_train)

model_mult_cat.coef_
model_mult_cat.intercept_

#E(LA|X)=b0+b1*LAT+b2*semi+b3*rural +b4(LAT*sermiurbano)+b5(LAT*rural)

## semi ur=0 ; rural =1

#E(LA|X)=b0+b1*LAT+b3 +b5*LAT= (b0+b3)+ LAT*(b1+b5)

X_train['interaccion_LATSemiurban'] = X_train["Semiurban"] * X_train['Loan_Amount_Term']

X_train['interaccion_LATRural'] = X_train['Rural'] * X_train['Loan_Amount_Term']



X_train_modelo_INT = X_train[["Loan_Amount_Term", "Semiurban", "Rural",'interaccion_LATSemiurban','interaccion_LATRural']]


modelo_lineal_interaccion = LinearRegression(fit_intercept=True)
modelo_lineal_interaccion.fit(X_train_modelo_INT,y_train)
modelo_lineal_interaccion.coef_
modelo_lineal_interaccion.intercept_

















