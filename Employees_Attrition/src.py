# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 08:56:12 2019

@author: marjan

A binary classifier to measure employees attrition rate

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def label_distribution(hr_df):
	"""
	Find label distribution
	"""
	no_label = hr_df[hr_df['Attrition']=='No'].shape[0]
	yes_label = hr_df[hr_df['Attrition']=='Yes'].shape[0]
	if yes_label < no_label:
		print "Data is imbalanced and 'yes' label is the rare-class."
	elif no_label < yes_label:
		print "Data is imbalanced and 'no' label is the rare-class."


def classifier(hr_df):
	"""
	"""
    ## check features types (numeric/ categoric) and convert categoric features to numeric
    ## collect all categorical features
	hr_df_cat = hr_df[['Attrition', 'BusinessTravel','Department',\
    'EducationField','Gender','JobRole','MaritalStatus','Over18', 'OverTime']].copy()
	Num_val = {'Yes':1, 'No':0}
	hr_df_cat['Attrition'] = hr_df_cat["Attrition"].apply(lambda x: Num_val[x])
	hr_df_cat = pd.get_dummies(hr_df_cat)

	hr_df_num = hr_df[['Age','DistanceFromHome', 'Education', 'EmployeeCount', 'EmployeeNumber',\
    'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',\
    'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',\
    'StandardHours', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',\
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']].copy()

    ## concatenate categorical and numerical features
	hr_df_final = pd.concat([hr_df_num, hr_df_cat], axis=1)

	print "Model building with random forest"

	target = hr_df_final['Attrition']
	features = hr_df_final.drop('Attrition', axis = 1)
    #create the train/test split
	X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=10)
    #Create the model and train
	model = RandomForestClassifier()
	model.fit(X_train,y_train)

	feat_importances = pd.Series(model.feature_importances_, index=features.columns)
	feat_importances = feat_importances.nlargest(20)
	print "Important features in employees attrition: ", feat_importances

	test_pred = model.predict(X_test)
    #test the accuracy
	acc = accuracy_score(y_test, test_pred)
	return acc





if __name__ == '__main__':

    path1 = 'input_data.csv'
    hr_df = pd.read_csv(path1)
    ## remove null values
    hr_df = hr_df.dropna()
    acc = classifier(hr_df)


