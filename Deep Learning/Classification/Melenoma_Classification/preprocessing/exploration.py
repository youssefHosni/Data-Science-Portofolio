# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 04:05:02 2021

@author: youss
"""
import pandas as pd
import matplotlib.pyplot as plt

def bar_plot(input_data,title):
    input_data.columns
    df=pd.DataFrame()
    df["disease"]=input_data.columns
    number_cases_per_class=input_data.sum()
    df["number_of_cases"]=number_cases_per_class.values
    plt.figure()
    plt.bar(df["disease"],df["number_of_cases"])
    plt.title(title)
    
def class_counts_proportions(labels):
    df=pd.DataFrame()
    df["Label"]=labels.columns
    number_cases_per_class=labels.sum()
    df["number_of_cases_each_class"]=number_cases_per_class.values
    df["percentage_of_classes"]=number_cases_per_class.values/sum(number_cases_per_class.values)   
    return df
