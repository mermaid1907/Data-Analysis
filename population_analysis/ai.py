# -*- coding: utf-8 -*-
from sklearn.linear_model import LinearRegression
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
#Connection Between Fertility Rate and Medium Age
plt.scatter(dataset["Med.Age"], dataset["Fert. Rate"])
plt.title("Connection Between Fertility Rate and Medium Age")
plt.xlabel("Med.Age")
plt.ylabel("Fert. Rate")
plt.savefig("fertility_mediumage.jpg")
plt.close()

#Connection Between Fertility Rate and Urban Population
plt.scatter(dataset["Urban Pop %"], dataset["Fert. Rate"])
plt.title("Connection Between Fertility Rate and Urban Population")
plt.xlabel("Urban Pop %")
plt.ylabel("Fert. Rate")
plt.savefig("urbanpopulation_fertility.jpg")
plt.close()

#Change of Population from Density(People/Km2)
plt.scatter(dataset["Density (P/Km²)"], dataset["Yearly change"])
plt.title("Change of Population from Density(People/Km2)")
plt.xlabel("Density (P/Km²)")
plt.ylabel("Yearly change")
plt.savefig("populaiton_density.jpg")
plt.close()


###################### NaN's ###########################

print("The rows that contain NaN's")
empty_rows = dataset.isna().sum()
print(empty_rows)

#using KNN algorithm to fill the NaN's
imputer = KNNImputer(n_neighbors=3)
dataset = imputer.fit_transform(dataset)
print("After Operation /n", dataset)


'''
###################### LINEAR REGRESSION ###########################


lr = LinearRegression()
model = lr.fit(dataset["Population (2022)"].values.reshape(-1, 1), dataset["Med.Age"] )

print(model.intercept_) #y-intercept (regression constant)
print(model.coef_) #regression coefficient, slope
'''



