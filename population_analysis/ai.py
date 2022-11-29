# -*- coding: utf-8 -*-
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


dataset = pd.read_csv("World_Population_2022_.csv")

#Columns: 
#[#, Country (or dependency), Population (2022), Yearly change, 
#Net change, Density (P/Km²), Land Area (Km²), Migrants (net), Fert. Rate, 
#Med.Age, Urban Pop %, World Share]

country_list = []
population_list = []
yearly_change_list = []
net_change_list = []
density_list = []
land_area_list = []
migrants_list = []
fertility_list = []
med_age_list = []
urb_pop_list = []
share_pop_list = []

for _, country, population, yearly_change, net_change, density, land_area, migrants, fertility, med_age, urb_pop, share_pop in dataset.values:
    country_list.append(str(country))
    population_list.append(int(population.replace(",", "")))
    yearly_change_list.append(float(yearly_change.replace("%", "")) / 100)
    net_change_list.append(int(net_change.replace(",", "")))
    density_list.append(int(density.replace(",", "")))
    land_area_list.append(int(land_area.replace(",", "")))
    migrants_list.append(None if str(migrants) == "nan" else int(str(migrants).replace(",", "")))
    fertility_list.append(None if fertility == "N.A." else float(fertility))
    med_age_list.append(None if med_age == "N.A." else int(med_age))
    urb_pop_list.append(None if urb_pop == "N.A." else float(urb_pop.replace("%", "")) / 100)
    share_pop_list.append(float(share_pop.replace("%", "")))
    
   
dataset["Country (or dependency)"] = country_list #Ülke isimleri
dataset["Population (2022)"] = population_list #Popülasyon
dataset["Yearly change"] = yearly_change_list #Yıllık değişim oranı
dataset["Net change"] = net_change_list #Net yıllık değişim
dataset["Density (P/Km²)"] = density_list #Kilometre başına düşen kişi sayısı
dataset["Land Area (Km²)"] = land_area_list #Ülkenin kağladığı alan km2
dataset["Migrants (net)"] = migrants_list #Net göç alma + net göç verme -
dataset["Fert. Rate"] = fertility_list #Doğum oranı
dataset["Med.Age"] = med_age_list #Ortanca yaş
dataset["Urban Pop %"] = urb_pop_list #Şehir dışı popülasyon
dataset["World Share"] = share_pop_list #Dünya'da paylaşılan popülasyon

print(dataset.describe())
print(dataset.dtypes)


###################### GRAPHS ###########################

#PIE CHARTS 
#Top Ten Population Sharing Countries
dataset = dataset.sort_values("World Share", ascending=False)
plt.pie(dataset["World Share"].head(10), labels = dataset["Country (or dependency)"].head(10))
plt.title("Top Ten Population Sharing Countries")
plt.savefig("worldshare.jpg")
plt.close()

#Top Ten Land Area Sharing Countries
dataset = dataset.sort_values("Land Area (Km²)", ascending=False)
plt.pie(dataset["Land Area (Km²)"].head(10), labels = dataset["Country (or dependency)"].head(10))
plt.title("Top Ten Land Area Sharing Countries")
plt.savefig("top10landarea.jpg")
plt.close()

#Top Ten Density (Population/Land Area) Countries
dataset = dataset.sort_values("Density (P/Km²)", ascending=False)
plt.pie(dataset["Density (P/Km²)"].head(10), labels = dataset["Country (or dependency)"].head(10))
plt.title("Top Ten Density (Population/Land Area) Countries")
plt.savefig("top10density.jpg")
plt.close()

#SCATTER GRAPHS
#Connection Between Fertility Rate and Urban Population
plt.scatter(dataset["Urban Pop %"], dataset["Fert. Rate"], color="pink")
plt.title("Connection Between Fertility Rate and Urban Population")
plt.xlabel("Urban Pop %")
plt.ylabel("Fert. Rate")
plt.savefig("urbanpopulation_fertility.jpg")
plt.close()

#Connection Between Fertility Rate and Medium Age
plt.scatter(dataset["Med.Age"], dataset["Fert. Rate"], color="red")
plt.title("Connection Between Fertility Rate and Medium Age")
plt.xlabel("Med.Age")
plt.ylabel("Fert. Rate")
plt.savefig("fertility_mediumage.jpg")
plt.close()

#Change of Population from Density(People/Km2)
plt.scatter(dataset["Density (P/Km²)"], dataset["Yearly change"])
plt.title("Change of Population from Density(People/Km2)")
plt.xlabel("Density (P/Km²)")
plt.ylabel("Yearly change")
plt.savefig("populaiton_density.jpg")
plt.close()

#Connection Between Fertility Rate and Yearly Change
plt.scatter(dataset["Yearly change"], dataset["Fert. Rate"])
plt.title("Connection Between Fertility Rate and Medium Age")
plt.xlabel("Med.Age")
plt.ylabel("Fert. Rate")
plt.savefig("yearly_fertility.jpg")
plt.close()



###################### NaN's ###########################
#fertility rate: 34 migrants: 34 med.age: 34 urban pop:13 

print("The rows that contain NaN's")
empty_rows = dataset.isna().sum()
print(empty_rows)

#using KNN algorithm to fill the NaN's
#imputer = KNNImputer(missing_values=None, strategy="mean")
#knn = imputer.fit_transform(dataset["Yearly change"].values.reshape(-1,1), dataset["Fert. Rate"])

#using fillna() to replace all the None values with the mean of that column
dataset['Migrants (net)'].fillna(dataset['Migrants (net)'].mean(), inplace=True)
dataset['Fert. Rate'].fillna(dataset['Fert. Rate'].mean(), inplace=True)
dataset['Med.Age'].fillna(dataset['Med.Age'].mean(), inplace=True)
dataset['Urban Pop %'].fillna(dataset['Urban Pop %'].mean(), inplace=True)


###################### LINEAR REGRESSION ###########################


lr = LinearRegression()
model = lr.fit(dataset["Fert. Rate"].values.reshape(-1, 1), dataset["Med.Age"] )

print(model.intercept_) #y-intercept (regression constant)
print(model.coef_) #regression coefficient, slope


#y_pred = 90*model.coef_ + model.intercept_
y_pred = model.predict(dataset["Fert. Rate"].values.reshape(-1,1))


#R squared -coefficient of determination
# %77 of the variation in fertility rate is explained by its medium age 
print(model.score(dataset["Fert. Rate"].values.reshape(-1, 1), dataset["Med.Age"]))

dataset.corr(numeric_only=True)

plt.plot(dataset["Fert. Rate"], dataset["Med.Age"], "o", color="green")
plt.plot(dataset["Fert. Rate"], y_pred, color="orange")
plt.title("Fertilitiy Rate by Medium Age")
plt.xlabel("Fertility Rate")
plt.ylabel("Medium Age")
plt.savefig("lr1")


#MIN MAX SCALER

min_max_scaler = preprocessing.MinMaxScaler()

dataset["Fert. Rate"] = pd.DataFrame(min_max_scaler
                                        .fit_transform(pd.DataFrame(dataset["Fert. Rate"])))

dataset["Urban Pop %"] = pd.DataFrame(min_max_scaler
                                        .fit_transform(pd.DataFrame(dataset["Urban Pop %"])))

first_country_list=[]
for _,_,_,_,_,_,_,_,fertility,_,urbpop,_ in dataset.values:
    
    first_country_list.append(True if urbpop > 0.7 and fertility < 0.4 else False)

dataset["First Country"] = first_country_list


#CONFUSION MATRIX

#normalization
y = dataset["First Country"]

x_data = dataset.drop(["Country (or dependency)", "First Country"], axis=1)
print(x_data)
x= (x_data-np.min(x_data))/(np.max(x_data) - np.min(x_data))

#train test split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=15)
print(x_train)
#random forest score
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
print(rfc.score(x_test,y_test))

#creating confusion matrix
y_pred=rfc.predict(x_test)
y_true=y_test

cm=confusion_matrix(y_true,y_pred)
print(cm)

#visualising confusion matrix

figure, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.title("Confusion Matrix of Yearly Change")
plt.xlabel("Real First Class Country")
plt.ylabel("Predict First Class Country")
plt.show() 


#Calculation of precision, recall, f1_score and accuracy
tp = cm[0,0]
fp = cm[1,0]
fn = cm[0,1]
tn = cm[1,1]

precision = tp / (tp+fp)
recall = tp / (tp+fn)
f1_score = 2 *(precision * recall)/(precision + recall)
accuracy = (tp + tn)/(tp + tn + fp + fn)

print("Precision: {}\nRecall: {}\nF1 Score: {}\nAccuracy: {}"
      .format(precision, recall, f1_score, accuracy))
