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

#-------------------------------------------------------------------------------------------

# DATA LOAD

#-------------------------------------------------------------------------------------------


# Load the Iris dataset.

  
df = pd.read_csv("iris.csv")
print (df)

#-------------------------------------------------------------------------------------------

# SUMMARY ANALYSIS

#-------------------------------------------------------------------------------------------

#Creating and opening a new text file for the summary.
with open("irissummary.txt", "wt") as file:  

     # Author.
    file.write("Author: Christiano Ferreira\n\n\n")

    # Writing the introduction.
    file.write("Higher Diploma in Science in Computing - Data Analytics\n\n")
    file.write("Programming and Scripting - Fisher's Iris\n\n")
    file.write("This project centers on accessing and analyzing the Iris dataset, a well-known dataset frequently employed in data science and\n")
    file.write("machine learning. Through this dataset, we aim to demonstrate a variety of skills, including data manipulation, statistical analysis,\n")
    file.write("visualization, and programming techniques using Python within the Jupyter Notebook environment.\n")
    file.write("The Iris flower dataset is a multivariate dataset introduced by British statistician and biologist Ronald Fisher in his 1936 paper,\n")
    file.write("\"The use of multiple measurements in taxonomic problems.\" It is also referred to as Anderson's Iris dataset because Edgar Anderson\n")
    file.write("collected the data to measure the morphological variation of Iris flowers from three related species: Setosa, Versicolor, and Virginica.\n")

    # Writing the Load Data.

    file.write("DATA LOAD:\n\n\n")

    file.write("Load the dataset into a DataFrame.\n\n")
    file.write("df = pd.read_csv('iris.csv')\n")
    file.write("print (df)\n\n")
    file.write(str(df) + "\n\n\n")

    file.write("The top 5 rows of the dataset.\n\n")
    file.write("5rows = df.head()\n")
    file.write("print (5rows)\n\n")
    file.write(str(df.head ()) + "\n\n")

    file.write("The last 5 rows of the dataset.\n\n")
    file.write("last5rows = df.tail()\n")
    file.write("print (lastrows)\n\n")
    file.write(str(df.tail ()) + "\n\n")

    file.write("Information of the dataset.\n\n")
    file.write("info = df.info ()\n")
    file.write("print (info)\n\n")
    file.write(str(df.info(buf=file)) + "\n\n") #[2]

    file.write("Count the number of null.\n\n")
    file.write("isnull = df.isnull().sum()\n")
    file.write("print (isnull)\n\n")
    file.write(str(df.isnull().sum()) + "\n\n")

    file.write("Describe the dataset.\n\n")
    file.write("describe = df.describe()\n")
    file.write("print (describe)\n\n")
    file.write(str(df.describe()) + "\n\n")  
    
    file.write("Compute the Pearson correlation.\n\n")
    file.write("correlation = df.corr(method='pearson', numeric_only=True)\n")
    file.write("print (correlation)\n\n")
    file.write(str(df.corr(method='pearson', numeric_only=True)) + "\n\n")   
    
    file.write("Extract the data from the \"species\" column.\n\n")
    file.write("df.value_counts(['species'])\n")
    file.write("print (species)\n\n")
    file.write(str(df.value_counts('species')) + "\n\n")
    
    file.write("Check the size of your DataFrame.\n\n")
    file.write("df.shape\n")
    file.write(str(df.shape) + "\n\n")

    file.write("Describe the data set by Species.\n\n")
    file.write("describe_species = df.groupby('species').describe().transpose()\n")
    file.write("print (describe_species)\n\n")
    file.write(str(df.groupby('species').describe().transpose()) + "\n\n")
    
    # Writing Visualization Data.

    file.write("DATA VISUALIZATION\n\n\n")

    file.write("Bar Chart\n\n")
    file.write("Fig. 01 - # Displaying the bar chart\n\n")

    file.write("Correlation\n\n")
    file.write("Fig. 02 - Correlation between Sepal Length and Width\n\n")

    file.write("Import the Iris dataset using Seaborn\n\n")
    file.write("Fig. 03 - Scatterplot of Sepal Length vs Sepal Width categorized by Species\n\n")

    file.write("Histogram\n\n")
    file.write("Figs. 04, 05, 06, 07 - Histogram of Sepal Length\n\n")

    file.write("Simple plot\n\n")
    file.write("Fig. 08 - Iris Dataset\n\n")

    file.write("Import the Iris dataset using Seaborn\n\n")
    file.write("Fig. 09 - Boxplot of Sepal Length categorized by Species\n\n")

    file.write("Create a heatmap\n\n")
    file.write("Fig. 10 - Heatmap of Correlations by Species\n\n")
    
    file.write("Pair plot to illustrate relationships between variables\n\n")
    file.write("Fig. 11 - Relationships among variables\n\n")

    # Writing End.

    file.write("End\n")

    file.write("Last update on 20/05/2024.\n\n")
    

#-------------------------------------------------------------------------------------------

# REFERENCES

#-------------------------------------------------------------------------------------------

# GeeksforGeeks. "Reading and Writing to Text Files in Python." Available at: https://www.geeksforgeeks.org/reading-writing-text-files-python/ [Accessed 20 May 2024].

#-------------------------------------------------------------------------------------------

# Last updated: 20/05/2024.
