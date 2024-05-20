#-------------------------------------------------------------------------------------------

#Higher Diploma in Science in Computing - Data Analytics

#-------------------------------------------------------------------------------------------

#Author: Christiano Ferreira

#-------------------------------------------------------------------------------------------

#Programming and Scripting - Fischer's Iris.

#This project centers on accessing and analyzing the Iris dataset, a well-known dataset frequently employed in data science and machine learning . 
#Through this dataset, we aim to demonstrate a variety of skills, including data manipulation, statistical analysis, visualization, and programming techniques using Python within the Jupyter Notebook environment .
#The Iris flower dataset is a multivariate dataset introduced by British statistician and biologist Ronald Fisher in his 1936 paper, "The use of multiple measurements in taxonomic problems." 
#It is also referred to as Anderson's Iris dataset because Edgar Anderson collected the data to measure the morphological variation of Iris flowers from three related species: Setosa, Versicolor and Virginica.

#-------------------------------------------------------------------------------------------

# LIBRARIES

#-------------------------------------------------------------------------------------------
 
# Data frames.
import pandas as pd

#-------------------------------------------------------------------------------------------

# DATA LOAD

#-------------------------------------------------------------------------------------------

#---------------------------
# Load the Iris dataset.
#---------------------------
  
df = pd.read_csv("iris.csv")
print (df)

#-------------------------------------------------------------------------------------------

# SUMMARY ANALYSIS

#-------------------------------------------------------------------------------------------
