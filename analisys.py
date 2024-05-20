#-------------------------------------------------------------------------------------------

#Higher Diploma in Science in Computing - Data Analytics

#-------------------------------------------------------------------------------------------

#Author: Christiano Ferreira

#-------------------------------------------------------------------------------------------

#Programming and Scripting - Fischer's Iris.

#   This project centers on accessing and analyzing the Iris dataset, a well-known dataset frequently employed in data science and machine learning . 
#Through this dataset, we aim to demonstrate a variety of skills, including data manipulation, statistical analysis, visualization, and programming techniques using Python within the Jupyter Notebook environment .
#   The Iris flower dataset is a multivariate dataset introduced by British statistician and biologist Ronald Fisher in his 1936 paper, "The use of multiple measurements in taxonomic problems." 
#It is also referred to as Anderson's Iris dataset because Edgar Anderson collected the data to measure the morphological variation of Iris flowers from three related species: Setosa, Versicolor and Virginica.

#-------------------------------------------------------------------------------------------

# LIBRARIES

#-------------------------------------------------------------------------------------------

# Data frames.
import pandas as pd

# Data visualization library.
import seaborn as sns

# Plotting.
import matplotlib.pyplot as plt

# Numerical arrays ad random numbers.
import numpy as np

#-------------------------------------------------------------------------------------------

# DATA LOAD

#-------------------------------------------------------------------------------------------


# Load the dataset into a DataFrame.

  
df = pd.read_csv("iris.csv")
print (df)


#      | sepal_length | sepal_width | petal_length | petal_width |     species
# -----|--------------|-------------|--------------|-------------|----------------
# 0    |          5.1 |       3.5   |        1.4   |      0.2    |   Iris-setosa
# 1    |          4.9 |       3.0   |        1.4   |      0.2    |   Iris-setosa
# 2    |          4.7 |       3.2   |        1.3   |      0.2    |   Iris-setosa
# 3    |          4.6 |       3.1   |        1.5   |      0.2    |   Iris-setosa
# 4    |          5.0 |       3.6   |        1.4   |      0.2    |   Iris-setosa
# ..   |          .   |       ...   |        ...   |      ...    |    ...
# 145  |          6.7 |       3.0   |        5.2   |      2.3    | Iris-virginica
# 146  |          6.3 |       2.5   |        5.0   |      1.9    | Iris-virginica
# 147  |          6.5 |       3.0   |        5.2   |      2.0    | Iris-virginica
# 148  |          6.2 |       3.4   |        5.4   |      2.3    | Iris-virginica
# 149  |          5.9 |       3.0   |        5.1   |      1.8    | Iris-virginica
#------|--------------|-------------|--------------|-------------|----------------
# [150 rows x 5 columns]

#-------------------------------------------------------------------------------------------

# INSPECT DATA

#-------------------------------------------------------------------------------------------


# The top 5 rows of the dataset.

first5 = df.head ()
print (first5)

#      | sepal_length | sepal_width | petal_length | petal_width |   species
# -----|--------------|-------------|--------------|-------------|-------------
#   0  |     5.1      |     3.5     |      1.4     |     0.2     | Iris-setosa	
#   1  |     4.9      |     3.0	    |      1.4	   |     0.2	 | Iris-setosa
#   2  |     4.7      |     3.2	    |      1.3	   |     0.2     | Iris-setosa
#   3  |     4.6      |     3.1	    |      1.5	   |     0.2	 | Iris-setosa
#   4  |     5.0      |     3.6	    |      1.4	   |     0.2	 | Iris-setosa
#------|--------------|-------------|--------------|-------------|-------------


# The last 5 rows of the dataset.

last5 = df.tail ()
print (last5)

#      | sepal_length | sepal_width | petal_length | petal_width |    species
# -----|--------------|-------------|--------------|-------------|----------------
#  145 |       6.7    |     3.0     |      5.2     |     2.3     | Iris-virginica
#  146 |       6.3    |     2.5     |      5.0     |     1.9     | Iris-virginica
#  147 |       6.5    |     3.0     |      5.2     |     2.0     | Iris-virginica
#  148 |       6.2    |     3.4     |      5.4     |     2.3     | Iris-virginica
#  149 |       5.9    |     3.0     |      5.1     |     1.8     | Iris-virginica
#------|--------------|-------------|--------------|-------------|----------------


# Information of the dataset.

info = df.info ()
print (info)

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 150 entries, 0 to 149
# Data columns (total 5 columns):
#        Column     |   Non-Null Count  |    Dtype  
#-------------------|-------------------|-------------
#  0   sepal_length |   150 non-null    |   float64
#  1   sepal_width  |   150 non-null    |   float64
#  2   petal_length |   150 non-null    |   float64
#  3   petal_width  |   150 non-null    |   float64
#  4   species      |   150 non-null    |   object 
#-----------------------------------------------------
# dtypes: float64(4), object(1)


# Count the number of null.

isnull = df.isnull().sum()
print (isnull)

#     Species    | Freq
# ---------------|------
#  sepal_length  |  0
#  sepal_width   |  0
#  petal_length  |  0
#  petal_width   |  0
#  species       |  0
# ---------------|------
# dtype: int64


# Describe the dataset.

describe = df.describe()
print (describe)

#        | sepal_length | sepal_width | petal_length | petal_width 
# -------|--------------|-------------|--------------|-------------
#  count |  150.000000	|  150.000000 |	150.000000   |  150.000000
#  mean	 |  5.843333	|  3.057333   |  3.758000    |   1.199333
#  std	 |  0.828066	|  0.435866   |  1.765298    |   0.762238
#  min	 |  4.300000	|  2.000000   |  1.000000    |   0.100000
#  25%	 |  5.100000	|  2.800000   |  1.600000    |   0.300000
#  50%	 |  5.800000	|  3.000000   |  4.350000    |   1.300000
#  75%	 |  6.400000	|  3.300000   |  5.100000    |   1.800000
#  max	 |  7.900000	|  4.400000   |  6.900000    |   2.500000
#--------|--------------|-------------|--------------|-------------


# Compute the Pearson correlation.

describe_species = df.groupby('species').describe().transpose()
print (describe_species)

#              | species | Iris-setosa | Iris-versicolor |	Iris-virginica
#--------------|---------|-------------|-----------------|-------------------
# sepal_length | count   | 50.000000   |   50.000000     |   50.000000
#              | mean    | 5.006000    |   5.936000      |   6.588000
#              | std     | 0.352490    |   0.516171      |   0.635880
#              | min     | 4.300000    |   4.900000      |   4.900000
#              | 25%     | 4.800000    |   5.600000      |   6.225000
#              | 50%     | 5.000000    |   5.900000      |   6.500000
#              | 75%     | 5.200000    |   6.300000      |   6.900000
#              | max     | 5.800000    |   7.000000      |   7.900000
#--------------|---------|-------------|-----------------|-------------------
# sepal_width  | count   | 50.000000   |   50.000000     |   50.000000
#              | mean    | 3.418000    |   2.770000      |   2.974000
#              | std     | 0.381024    |   0.313798      |   0.322497
#              | min     | 2.300000    |   2.000000      |   2.200000
#              | 25%     | 3.125000    |   2.525000      |   2.800000
#              | 50%     | 3.400000    |   2.800000      |   3.000000
#              | 75%     | 3.675000    |   3.000000      |   3.175000
#              | max     | 4.400000    |   3.400000      |   3.800000
#--------------|---------|-------------|-----------------|-------------------
# petal_length | count   | 50.000000   |   50.000000     |   50.000000
#              | mean    | 1.464000    |   4.260000      |   5.552000
#              | std     | 0.173511    |   0.469911      |   0.551895
#              | min     | 1.000000    |   3.000000      |   4.500000
#              | 25%     | 1.400000    |   4.000000      |   5.100000
#              | 50%     | 1.500000    |   4.350000      |   5.550000
#              | 75%     | 1.575000    |   4.600000      |   5.875000
#              | max     | 1.900000    |   5.100000      |   6.900000
#--------------|---------|-------------|-----------------|-------------------
# petal_width  | count   | 50.000000   |   50.000000     |   50.000000
#              | mean    | 0.244000    |   1.326000      |   2.026000
#              | std     | 0.107210    |   0.197753      |   0.274650
#              | min     | 0.100000    |   1.000000      |   1.400000
#              | 25%     | 0.200000    |   1.200000      |   1.800000
#              | 50%     | 0.200000    |   1.300000      |   2.000000
#              | 75%     | 0.300000    |   1.500000      |   2.300000
#              | max     | 0.600000    |   1.800000      |   2.500000
#--------------|---------|-------------|-----------------|-------------------


# Extract the data from the \"species\" column.

value_counts = df.value_counts(['species'])
print (value_counts)

#      species      |  Freq
# ------------------|---------
#       setosa      |   50
#     versicolor    |   50
#     virginica     |   50
# ----------------------------
# Name: count, dtype: int64


# Check the size of your DataFrame.

correlation = df.corr(method='pearson', numeric_only=True)
print (correlation)

#                | sepal_length | sepal_width | petal_length | petal_width 
# ---------------|--------------|-------------|--------------|-------------
#  sepal_length	 |   1.000000   |  -0.117570  |    0.871754  |   0.817941
#  sepal_width   |  -0.117570   |   1.000000  |   -0.428440  |  -0.366126
#  petal_length  |   0.871754   |  -0.428440  |    1.000000  |   0.962865
#  petal_width   |   0.817941   |  -0.366126  |    0.962865  |   1.000000
#----------------|--------------|-------------|--------------|-------------


# Describe the data set by Species.

correlation_species = df.groupby('species').corr(method='pearson', numeric_only=True)
print (correlation_species)

#     species     |              | sepal_length | sepal_width | petal_length | petal_width 
# ----------------|--------------|--------------|-------------|--------------|---------------					
#    Iris-setosa  | sepal_length |   1.000000   |   0.742547  |   0.267176   |   0.278098
#                 | sepal_width  |   0.742547   |   1.000000  |   0.177700   |   0.232752
#                 | petal_length |   0.267176   |   0.177700  |   1.000000   |   0.331630
#                 | petal_width  |   0.278098   |   0.232752  |   0.331630   |   1.000000
#-----------------|--------------|--------------|-------------|--------------|---------------
# Iris-versicolor | sepal_length |   1.000000   |   0.525911  |   0.754049   |   0.546461
#                 | sepal_width  |   0.525911   |   1.000000  |   0.560522   |   0.663999
#                 | petal_length |   0.754049   |   0.560522  |   1.000000   |   0.786668
#                 | petal_width  |   0.546461   |   0.663999  |   0.786668   |   1.000000
#-----------------|--------------|--------------|-------------|--------------|---------------
# Iris-virginica  | sepal_length |   1.000000   |   0.457228  |   0.864225   |   0.281108
#                 | sepal_width  |   0.457228   |   1.000000  |   0.401045   |   0.537728
#                 | petal_length |   0.864225   |   0.401045  |   1.000000   |   0.322108
#                 | petal_width  |   0.281108   |   0.537728  |   0.322108   |   1.000000
#-----------------|--------------|--------------|-------------|------------------------------

#-------------------------------------------------------------------------------------------

# DATA VISUALIZATION

#-------------------------------------------------------------------------------------------

# Import the Iris dataset into a DataFrame from a URL
url = 'https://raw.githubusercontent.com/corpuschris/pands-project/master/iris.csv'
df = pd.read_csv(url)

#-------------------------------------------------------------------------------------------


# Determine the quantity of flowers for each species
species_counts = df['species'].value_counts()

# Import the Iris dataset into a DataFrame from a URL
url = 'https://raw.githubusercontent.com/corpuschris/pands-project/master/iris.csv'
df = pd.read_csv(url)

# Determine the quantity of flowers for each species
species_counts = df['species'].value_counts()

# Plotting bar chart
plt.bar(species_counts.index, species_counts.values, color=['#9370db', '#800080', '#a98ac9'])
plt.xlabel('Species')
plt.ylabel('Number of Flowers')
plt.title('Number of Iris Flowers by Species')
plt.show()

# Save as a PNG file
plt.savefig('species_counts_bar_chart.png')

#-------------------------------------------------------------------------------------------

# Correlation between Sepal Length and Width
# Create a figure with specified size.
plt.figure(figsize=(14,8))

# Plot scatter for Iris-setosa.
ax = df[df.species=='Iris-setosa'].plot.scatter(x='sepal_length', y='sepal_width', color='#9370db', label='Setosa')

# Plot scatter for Iris-versicolor.
df[df.species=='Iris-versicolor'].plot.scatter(x='sepal_length', y='sepal_width', color='#800080', label='Versicolor', ax=ax)

# Plot scatter for Iris-virginica.
df[df.species=='Iris-virginica'].plot.scatter(x='sepal_length', y='sepal_width', color='#a98ac9', label='Virginica', ax=ax)

# Set x-axis label.
ax.set_xlabel("Sepal Length")

# Set y-axis label.
ax.set_ylabel("Sepal Width")

# Set plot title.
ax.set_title("Sepal Length vs Width")

# Display the plot.
plt.show()


#-------------------------------------------------------------------------------------------

# Load the Iris dataset from Seaborn
iris = sns.load_dataset('iris')

# Define colors for each species
colors = {'setosa': '#9370db', 'versicolor': '#800080', 'virginica': '#a98ac9'}

# Create a scatterplot using Seaborn
sns.scatterplot(data=iris, x='sepal_length', y='sepal_width', hue='species', style='species', palette=colors)

# Add a title
plt.title('Scatterplot of Sepal Length vs Sepal Width by Species')

# Save as a PNG file
plt.savefig('sepalscatterplot.png')

# Show the plot
plt.show()

#-------------------------------------------------------------------------------------------
# Histogram
# Define colors for each column
colors = {'sepal_length': '#9370db', 'sepal_width': '#800080', 'petal_length': '#a98ac9', 'petal_width': '#4b0082'}

# Generate histograms and save them as PNG files
for column, color in colors.items():
    # Create a new figure for each histogram
    plt.figure()  

    # Create histogram for column with specified properties
    plt.hist(df[column], color=color, edgecolor='black', linewidth=1)

    # Set title for the histogram
    plt.title(f'Histogram of {column}')

    # Set xlabel for the histogram
    plt.xlabel('Values')  

    # Set ylabel for the histogram
    plt.ylabel('Frequency')  

    # Save the histogram as a PNG file
    plt.savefig(f'{column}_histogram.png', bbox_inches='tight')  

    # Show the histogram
    plt.show()  

#-------------------------------------------------------------------------------------------

# ### Two Variable Plots:
# ***


# Extracting petal lengths
plen = df['petal_length']

# Displaying
print(plen)

# Type
print(type(plen))

# Extracting as numpy array
plen = plen.to_numpy()

# Displaying
plen

# Petal widths
pwidth = df['petal_width'].to_numpy()

# Show
pwidth

#-------------------------------------------------------------------------------------------

# Simple plot
plt.plot(plen, pwidth, 'x', color='#800080')

# Axis labels
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')

# Title
plt.title('Petal Length vs Width')

# X limits
plt.xlim(0, 8)

# Y limits
plt.ylim(0, 4)

# Save as a PNG file
plt.savefig('simple_plot.png')

#-------------------------------------------------------------------------------------------

# Load the Iris dataset using Seaborn
iris = sns.load_dataset('iris')

# Define a custom color palette
custom_palette = ['#9370db', '#800080', '#a98ac9']

# Create a boxplot using Seaborn with hue for species differentiation and the specified palette
sns.boxplot(data=iris, x='species', y='sepal_length', hue='species', palette=custom_palette)

# Add a title to the plot
plt.title('Sepal Length Distribution by Species')

# Show the plot
plt.show()

# Save as a PNG file
plt.savefig('seaborn_boxplot.png')

#-------------------------------------------------------------------------------------------

# Create a correlation matrix
correlation_matrix = df.groupby('species').corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Generate  heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='Purples', fmt=".2f")

# Choose title
plt.title('Correlation Heatmap by Species')

# Display the plot
plt.show()

# Save as a PNG file
plt.savefig('correlation_heatmap.png')

#-------------------------------------------------------------------------------------------

# Generate a pair plot to visualize relationships between variables
# - 'data=df': Specifies the DataFrame containing the dataset for visualization.
sns.pairplot(data=df, hue='species', height=3, palette='Purples')

# Retrieve unique species in the DataFrame
species_list = df['species'].unique()

# Save the as a PNG file
plt.savefig('pair_plot.png')

# Display the plot
plt.show()

#-------------------------------------------------------------------------------------------

# End
# Last updated: 20/05/2024.


