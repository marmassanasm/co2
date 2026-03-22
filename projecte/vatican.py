# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:18:47 2024

@author: Usuario
"""

import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv(r'C:\Users\Usuario\Downloads\owid-co2-data.csv')

# ll=['World','Asia','Europe','America','GCP','income']

# for i in ll:
#     data=data[~data['country'].str.contains(i)]

# data=data[data['year'] == 2022].sort_values(by='co2', ascending=False)[-20:]

# plt.bar(x=data['country'],height=data['co2'])



# plt.xticks(rotation=60)
# plt.show()


european_countries = [
    "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium",
    "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark",
    "Estonia", "Finland", "France", "Georgia", "Germany", "Greece", "Hungary", "Iceland",
    "Ireland", "Italy", "Kosovo", "Latvia", "Liechtenstein", "Lithuania",
    "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia",
    "Norway", "Poland", "Portugal", "Romania", "Russia", "San Marino", "Serbia", "Slovakia",
    "Slovenia", "Spain", "Sweden", "Switzerland", "Turkey", "Ukraine", "United Kingdom",
    "Vatican"
]


data=data[data['country'].str.contains('|'.join(european_countries))]

data=data[data['year'] == 2000].sort_values(by='co2', ascending=False)[-10:]

plt.bar(x=data['country'],height=data['co2'])



plt.xticks(rotation=60)
plt.show()

