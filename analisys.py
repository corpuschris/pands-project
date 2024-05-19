# -----------------------------------------------------------------------------------------------

# Higher Diploma in Science in Computing - Data Analytics

# -----------------------------------------------------------------------------------------------

# Programming and Scripting - Iris 

# * Setosa
# * Versicolor
# * Virginica

# -----------------------------------------------------------------------------------------------

# About this project:

# -----------------------------------------------------------------------------------------------

# Libraries:
 
# -----------------------------------------------------------------------------------------------

# Importing necessary libraries for data manipulation, visualization, and numerical operations:

# Data frames [3]
import pandas as pd

# Data visualization library [3]
import seaborn as sns

# Plotting [5]
import matplotlib.pyplot as plt

# Numerical arrays and random numbers [6]
import numpy as np

# -----------------------------------------------------------------------------------------------

# Load Data:

# -----------------------------------------------------------------------------------------------

# Load Iris Dataset:

# -----------------------------------------------------------------------------------------------

# Load the dataset into a DataFrame [8]
df = pd.read_csv("iris.csv")
print(df)

# -----------------------------------------------------------------------------------------------

#          sepal_length  sepal_width  petal_length  petal_width  species
# 0             5.1          3.5           1.4          0.2       setosa
# 1             4.9          3.0           1.4          0.2       setosa
# 2             4.7          3.2           1.3          0.2       setosa
# 3             4.6          3.1           1.5          0.2       setosa
# 4             5.0          3.6           1.4          0.2       setosa
# ..            ...          ...           ...          ...         ...
# 145           6.7          3.0           5.2          2.3       virginica
# 146           6.3          2.5           5.0          1.9       virginica
# 147           6.5          3.0           5.2          2.0       virginica
# 148           6.2          3.4           5.4          2.3       virginica
# 149           5.9          3.0           5.1          1.8       virginica

# -----------------------------------------------------------------------------------------------

# Inspect Data:

# -----------------------------------------------------------------------------------------------

# Display the first 5 rows of the dataset.
fiverows = df.head()
print(fiverows)

# -----------------------------------------------------------------------------------------------
# sepal_length    sepal_width    petal_length    petal_width    species
# 5.1             3.5            1.4             0.2            setosa
# 4.9             3.0            1.4             0.2            setosa
# 4.7             3.2            1.3             0.2            setosa
# 4.6             3.1            1.5             0.2            setosa
# 5.0             3.6            1.4             0.2            setosa

# -----------------------------------------------------------------------------------------------

# Display the last 5 rows of the dataset.
lastrows = df.tail()
print(lastrows)

# -----------------------------------------------------------------------------------------------

# sepal_length    sepal_width    petal_length    petal_width    species
# 145             6.7            3.0             5.2            2.3       virginica
# 146             6.3            2.5             5.0            1.9       virginica
# 147             6.5            3.0             5.2            2.0       virginica
# 148             6.2            3.4             5.4            2.3       virginica
# 149             5.9            3.0             5.1            1.8       virginica

# -----------------------------------------------------------------------------------------------

# Display information about the dataset.
inf = df.info()
print(inf)

# -----------------------------------------------------------------------------------------------

# <class 'pandas.core.frame.DataFrame'>
#  RangeIndex: 150 entries, 0 to 149
#  Data columns (total 5 columns):
#   Column        Non-Null Count  Dtype  
#  ---  ------        --------------  -----  
#   0   sepal_length  150 non-null    float64
#   1   sepal_width   150 non-null    float64
#   2   petal_length  150 non-null    float64
#   3   petal_width   150 non-null    float64
#   4   species       150 non-null    object 
#  dtypes: float64(4), object(1) 
#  memory usage: 6.0+ KB

# -----------------------------------------------------------------------------------------------
