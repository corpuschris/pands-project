# %% [markdown]
# ## Higher Diploma in Science in Computing - Data Analytics
# 
# ### Author: Christiano Ferreira
# ***
# ### Programming and Scripting - Fischer's Iris.
# 
# This project focuses on accessing and analyzing the [Iris](https://github.com/corpuschris/pands-project/blob/master/iris.csv) dataset, a classic dataset frequently used in data science and machine learning.[1] By working with this dataset, we aim to showcase a range of skills, including data manipulation, statistical analysis, visualization, and programming techniques using [Python](https://www.python.org/) within the Jupyter Notebook environment.[2]
# 
# The Iris flower dataset is a multivariate dataset introduced by the British statistician and biologist Ronald Fisher in his 1936 paper "The use of multiple measurements in taxonomic problems." It is also known as Anderson's Iris dataset because Edgar Anderson collected the data to measure the morphological variation of Iris flowers from three related species:
# 
# * Setosa
# * Versicolor
# * Virginica
# ***
# 
# ### About this project:
# 
# It is [Iris](https://github.com/corpuschris/pands-project/blob/master/iris.csv) dataset.
# 
# ![Iris](https://miro.medium.com/v2/resize:fit:720/1*YYiQed4kj_EZ2qfg_imDWA.png)
# 
# [Fischer's Iris Data](https://archive.ics.uci.edu/dataset/53/iris) UCI Machine Learning Repository: Iris Data Set.
# 
# 
# ***

# %% [markdown]
# ### Necessary Imports:
# ***
# 
# #### To handle, analyze, and visualize the data effectively, this project utilized several essential libraries:
# ***
# 
# * Pandas: A versatile and efficient open-source tool for data analysis and manipulation.
# * Numpy: Library primarily used for array operations, including linear algebra and matrix manipulations.
# * Matplotlib: Comprehensive library for creating various types of visualizations.
# * Seaborn: Data visualization library that enhances Matplotlib with high-level statistical graphics capabilities.

# %%
# Data frames.
import pandas as pd

# Data visualization library.
import seaborn as sns

# Plotting.[5]
import matplotlib.pyplot as plt

# Numerical arrays ad random numbers.
import numpy as np




# %% [markdown]
# ### Data Load:
# ***
# The dataset retrieved from the Iris Dataset has been stored in a file named iris.csv within the repository. This .csv file is accessed using the pandas library with the command [df = pd.read_csv](https://github.com/corpuschris/pands-project/blob/master/iris.csv)

# %%
# Load the dataset into a DataFrame
df = pd.read_csv("iris.csv")
print (df)

# %% [markdown]
# ### Inspect Data:
# ***
# 
# The *inspect data* process involves examining and understanding the structure, content, and characteristics of a dataset. It provides users with valuable insights to comprehend the dataset and make informed decisions about data preprocessing, analysis, and modeling. Here's what *inspect data* typically provides for users to understand the dataset:

# %%
# The top 5 rows of the dataset.
df.head ()

# %%
# The last 5 rows of the dataset.
df.tail ()

# %%
# Informations of the dataset.
df.info ()

# %%
# Count the number of null.
df.isnull ().sum ()

# %%
# Descibe the data set.
df.describe ()

# %%
# Compute the Pearson correlation.[14]
df.corr(method='pearson', numeric_only=True)

# %%
# Extract the data from the "species" column.
df.value_counts(['species'])

# %%
# Check the size of your DataFrame.
df.shape
print(df.shape)

# %%
# Describe the data set by Species.
describe_species = df.groupby('species').describe().transpose()
print (describe_species)

# %%
# Correlation of the data set by Species.
correlation_species = df.groupby('species').corr(method='pearson', numeric_only=True)
print (correlation_species)

# %% [markdown]
# ### Data Visualization:
# 
# 
# * Data visualization is the graphical representation of data to communicate information effectively and efficiently. It involves the use of visual elements such as charts, graphs, and maps to explore, analyze, and present data in a visual format.
# 
# ***

# %%
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

# %%
plt.figure(figsize=(14,8))
ax = df[df.species=='Iris-setosa'].plot.scatter(x='sepal_length', y='sepal_width', color='#9370db', label='Setosa')
df[df.species=='Iris-versicolor'].plot.scatter(x='sepal_length', y='sepal_width', color='#800080', label='Versicolor', ax=ax)
df[df.species=='Iris-virginica'].plot.scatter(x='sepal_length', y='sepal_width', color='#a98ac9', label='Virginica', ax=ax)
ax.set_xlabel("Sepal Length")
ax.set_ylabel("Sepal Width")
ax.set_title("Relationship between Sepal Length and Width")
plt.show()

# %%
# Load the Iris dataset from Seaborn
iris = sns.load_dataset('iris')

# Define colors for each species
colors = {'setosa': '#9370db', 'versicolor': '#800080', 'virginica': '#a98ac9'}

# Create a scatterplot using Seaborn
sns.scatterplot(data=iris, x='sepal_length', y='sepal_width', hue='species', style='species', palette=colors)

# Add a title
plt.title('Scatterplot of Sepal Length vs Sepal Width by Species')

# Show the plot
plt.show()

# %%
# Define colors for each column
colors = {'sepal_length': '#9370db', 'sepal_width': '#800080', 'petal_length': '#a98ac9', 'petal_width': '#4b0082'}

# Generate histograms and save them as PNG files
for column, color in colors.items():
    plt.figure()  # Create a new figure for each histogram
    plt.hist(df[column], color=color, edgecolor='black', linewidth=1)  # Create histogram for column with specified properties
    plt.title(f'Histogram of {column}')  # Set title for the histogram
    plt.xlabel('Values')  # Set xlabel for the histogram
    plt.ylabel('Frequency')  # Set ylabel for the histogram
    plt.savefig(f'{column}_histogram.png', bbox_inches='tight')  # Save the histogram as a PNG file
    plt.show()  # Show the histogram


# %% [markdown]
# ### Two Variable Plots:
# ***

# %%
# Extracting petal lengths
plen = df['petal_length']

# Displaying
print(plen)

# Type
print(type(plen))

# %%
# Extracting as numpy array
plen = plen.to_numpy()

# Displaying
plen


# %%
# Petal widths
pwidth = df['petal_width'].to_numpy()

# Show
pwidth

# %%
# Simple plot
plt.plot(plen, pwidth, 'x', color='#800080')

# Axis labels
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')

# Title
plt.title('Iris Data Set')

# X limits
plt.xlim(0, 8)

# Y limits
plt.ylim(0, 4)


# %%
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



# %%
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


# %%
# Generate a pair plot to illustrate relationships among variables
# - 'data=df': Specifies the DataFrame containing the dataset for visualization.
sns.pairplot(data=df, hue='species', height=3)

# Retrieve unique species in the DataFrame
species_list = df['species'].unique()

# Define the palette with various shades of purple
purple_palette = sns.color_palette("Purples")

# Show the figure.
plt.show()


# %%
# Load the Iris dataset from a URL into a DataFrame
url = 'https://raw.githubusercontent.com/corpuschris/pands-project/master/iris.csv'
df = pd.read_csv(url)

# Count the number of flowers for each species
species_counts = df['species'].value_counts()

# Plot the bar chart and save as PNG
plt.bar(species_counts.index, species_counts.values, color=['#9370db', '#800080', '#a98ac9'])
plt.xlabel('Species')
plt.ylabel('Number of Flowers')
plt.title('Number of Iris Flowers by Species')
plt.savefig('species_counts_bar_chart.png')
plt.show()

# Scatterplot of Sepal Length vs Sepal Width
plt.figure(figsize=(14,8))
ax = df[df.species=='Iris-setosa'].plot.scatter(x='sepal_length', y='sepal_width', color='#9370db', label='Setosa')
df[df.species=='Iris-versicolor'].plot.scatter(x='sepal_length', y='sepal_width', color='#800080', label='Versicolor', ax=ax)
df[df.species=='Iris-virginica'].plot.scatter(x='sepal_length', y='sepal_width', color='#a98ac9', label='Virginica', ax=ax)
ax.set_xlabel("Sepal Length")
ax.set_ylabel("Sepal Width")
ax.set_title("Relationship between Sepal Length and Width")
plt.savefig('sepal_length_vs_sepal_width_scatterplot.png')
plt.show()

# Create a scatterplot using Seaborn and save as PNG
iris = sns.load_dataset('iris')
colors = {'setosa': '#9370db', 'versicolor': '#800080', 'virginica': '#a98ac9'}
sns.scatterplot(data=iris, x='sepal_length', y='sepal_width', hue='species', style='species', palette=colors)
plt.title('Sepal Length vs Sepal Width by Species')
plt.savefig('seaborn_scatterplot.png')
plt.show()

# Histograms and save each as PNG
colors = {'sepal_length': '#9370db', 'sepal_width': '#800080', 'petal_length': '#a98ac9', 'petal_width': '#4b0082'}
for column, color in colors.items():
    plt.figure()
    plt.hist(df[column], color=color, edgecolor='black', linewidth=1)
    plt.title(f'Histogram of {column}')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.savefig(f'{column}_histogram.png', bbox_inches='tight')
    plt.show()

# Simple plot and save as PNG
plt.plot(df['petal_length'], df['petal_width'], 'x', color='#800080')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal Length vs Petal Width')
plt.xlim(0, 8)
plt.ylim(0, 4)
plt.savefig('simple_plot.png')
plt.show()

# Boxplot using Seaborn and save as PNG
color_palette = ['#9370db', '#800080', '#a98ac9']
sns.boxplot(data=iris, x='species', y='sepal_length', hue='species', palette=color_palette)
plt.title('Sepal Length by Species')
plt.savefig('seaborn_boxplot.png')
plt.show()

# Correlation Heatmap and save as PNG
correlation_matrix = df.groupby('species').corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Purples', fmt=".2f")
plt.title('Correlation Heatmap by Species')
plt.savefig('correlation_heatmap.png')
plt.show()

# Pair plot and save as PNG
sns.pairplot(data=df, hue='species', height=3)
species_list = df['species'].unique()
blue_palette = sns.color_palette("Purples")
plt.savefig('pair_plot.png')
plt.show()


# %% [markdown]
# ### References:
# 
# - UCI Machine Learning Repository: [Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris). Accessed on 14 May 2024.
# - Official pandas documentation for DataFrame methods: [Pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html). Accessed on 14 May 2024.
# - Matplotlib documentation on creating scatter plots: [matplotlib.pyplot.scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html). Accessed on 14 May 2024.
# - Stack Overflow: [A public platform of coding questions & answers](https://stackoverflow.com/). Accessed on 14 May 2024.
# - [Iris flower Data set](https://en.wikipedia.org/wiki/Iris_flower_data_set). Accessed on 14 May 2024.
# - Seaborn: [Introduction](https://seaborn.pydata.org/tutorial/introduction.html). Accessed on 14 May 2024.
# - Pandas `.head` method: [Pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html). Accessed on 14 May 2024.
# - Pandas `.tail` method: [Pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html). Accessed on 14 May 2024.
# - Pandas `.info` method: [Pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html). Accessed on 14 May 2024.
# - Pandas `.isnull` method: [Pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isnull.html). Accessed on 14 May 2024.
# - Pandas `.describe` method: [Pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html). Accessed on 14 May 2024.
# - Pandas `.corr` method: [Pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html). Accessed on 14 May 2024.
# - Pandas `.value_counts` method: [Pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html). Accessed on 14 May 2024.
# - Pandas `.shape` method: [Pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html). Accessed on 14 May 2024.
# - Pandas `.columns` method: [Pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.columns.html). Accessed on 14 May 2024.

# %% [markdown]
# 


