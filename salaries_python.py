# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 20:57:19 2023

@author: MD
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



filepath = 'C:/MD/Dokumenty/python/data_analysis/salaries_analysis/Salary_Data.csv'
filepath_cleaned = 'C:/MD/Dokumenty/python/data_analysis/salaries_analysis/Salary_Data_cleaned.csv'



def create_data_series_for_boxplots(gender):
    data = pd.DataFrame((df[(df['Gender'] == gender)]).reset_index())
    data = data[data.columns[6]]
    data.rename(gender, inplace = True)
    return data 




if __name__ == "__main__":
    
    
    # EDA
    
    # Read file
    df = pd.read_csv(filepath)
    
    # Explore data
    describe = df.describe() 
    info = df.info()
        
    # Check missing values
    missing_values = df.isnull().sum()
    
    # Dropping missing values
    df = df.dropna()
    
    # Merging same education levels
    edu_levels = df['Education Level'].unique()
    df['Education Level'] = df['Education Level'].replace("Bachelor's Degree", "Bachelor's")
    df['Education Level'] = df['Education Level'].replace("Master's Degree","Master's")
    df['Education Level'] = df['Education Level'].replace("phD","PhD")
    
    # Changing Age, Years of Experience and Salary columns' data type to integer
    df['Age'] = df['Age'].astype(int)
    df['Years of Experience'] = df['Years of Experience'].astype(int)
    df['Salary'] = df['Salary'].astype(int)
    info = df.info()
    
    # Saving cleaned data into new file
    df.to_csv(filepath_cleaned, index=False)
       
    # Create correlation matrix
    corr_matrix = df[['Age', 'Years of Experience', 'Salary']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="Blues")
    plt.title('Correlation matrix', fontsize=16)
    plt.show()
    
    
    # MODEL
    
    # Linear regression - Years of Experience and Salary
    lm = LinearRegression()
    x = df[['Years of Experience']]
    y = df[['Salary']]
    lm.fit(x,y)
    r_sql = lm.score(x,y)
    yhat = lm.predict(x)
    sns.regplot(data=df, x='Years of Experience', y='Salary')
    plt.title('Linear Regression: Salary and Years of Experience', fontsize=16)
    plt.show()
    sns.boxplot(x = 'Years of Experience', y = 'Salary', data = df)
    plt.title('Outliers of Years of Experience and Salary')
    plt.show()
    
    # Linear regression - Age and Salary
    x = df[['Age']]
    y = df[['Salary']]
    lm.fit(x,y)
    r_sql = lm.score(x,y)
    sns.regplot(data=df, x='Age', y='Salary')
    plt.title('Linear Regression: Salary and Age', fontsize=16)
    plt.show()
    sns.boxplot(x = 'Age', y = 'Salary', data = df)
    plt.title('Outliers of Age and Salary')
    plt.show()
    
    # Multivariate Regression
    z = df[['Years of Experience','Age']]
    y = df[['Salary']]
    lm.fit(z,y)
    r_sqm = lm.score(z,y)
    yhatm = lm.predict(z)
    
    
    # CONCLUSIONS
    # The independent variable explians only 53% (in case of linear regression)
    # and 66% (in case of multivariate regression) of dependent variable. 
    # This is because salaries vary significantly depending on the job title.
    # The correct model should be built separately for every job title,
    # so more data would be needed. Alternatively, the better regression model 
    # should be chosen.
    
    
    # BOXPLOT FOR POWERBI DASHBOARD
    
    # Salary by gender
    male = create_data_series_for_boxplots('Male')
    female = create_data_series_for_boxplots('Female')
    other = create_data_series_for_boxplots('Other')
    boxplots = pd.concat([male, female, other], axis = 1)
    ax = sns.boxplot(boxplots, palette='Blues')
    ax.set_title('Salary by gender', fontsize=16)
    ax.set_xlabel('Gender', fontsize=12)
    ax.set_ylabel('Salary', fontsize=12)

    
    
       