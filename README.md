Problem 1

A firm wants to understand why some of its best employees are leaving the company. The firm also wants to predict the employees who might leave next, understand the reasons as to why and what should they do to retain them. Please answer the following questions using machine learning techniques by referring to the attached synthetic data set that provides some information on 1470 employees:

(i) Identify the reasons that lead to employees leaving the firm. (ii) Predicts which employees (from the data set) might leave next.

(iii) Recommend which employees from point (ii) above should be retained? Output: Submission should contain the Model/Code and analysis of the problem


We need to detect the importance of features in the data that leads to employees attrition.
This is a binary classification problem. Classes are ‘yes’ or ‘no’ which means if the employee
will leave the company or not. The data set is imbalanced and the samples with ‘yes’ as the
label are much less than the samples with ‘no’ as the label. After removing null values and
transforming categorical features into numerical features, we need to choose an estimator.
For this problem, a random forest can make a good result because it works well with default
parameters and does not overfit easily. Also, it is a discriminative model that can work well
with a small amount of data. Also, it provides the importance of the variables in the data which
can be seen as the reasons that lead to employees attrition.
After implementing RandomForest on data, the most important features are as follow:
“Age”, “MonthlyIncome”, and “TotalWorkingHours”
However, it is better to remove highly correlated features to make model faster and get a
better result. It seems MonthlyIncome is the most important factor while the model detects it
as the second feature.

Generally, the employees with lower “MonthlyIncome” have a higher chance of leaving the
company. Other features such as “Age” and “TotalWorkingHours” should also be considered.
First, we need to check how these features affect the attrition rate. For example, if we can
decrease attrition by increasing “MonthlyIncome” or decreasing “MonthlyIncome”. In other
words, if the feature has a positive or a negative correlation to the target. Pandas library has
methods to measure correlations between features and between a feature and target.

Employees whose features determine their high value for the company should be retained.
These features can be the number of years that they have worked for the company, total work
experience, performance, education, and job involvement.



Problem 2

Please answer the following questions based on your understanding and experience working with various learning techniques:

(i)  You built a model that is ”too good to be true”, what could be the reasons for it and what would you do as the next steps?

(ii)  You are tasked to classify a rare-occurring class; which of the two class of models would you choose and why: Bayesian models or Regression-based models?

When a model works too well on a data set, it cannot predict well on unseen data. This
problem is called overfitting and it happens when a model learns all noise in the training data.
This problem affects model performance in generalizing on test data. Reasons for overfitting
can be a complex target function (that is customized for all details in the train set), lack of
training data, imbalanced data set.
To overcome this issue, we should see if we can collect more data, undersample or
oversample available data to make it balanced, implement cross_validation, use the
early-stopping technique, add regularization parameters to make target function simpler.

To deal with a rare_occurring class (imbalanced data set), we need to use undersampling
or oversampling methods. If data is small (less than 100k) oversampling from rare class can
be a good option. In this case, to avoid repetition, added samples should be audited in some
way to have unique data samples. If the data set is small, undersample the frequent_class
can be a better option to preserve the uniqueness of data samples. After the undersampling,
the data set size will decrease. When the data set is small a regression-based model
(discriminative classifier) can make better result in splitting samples based on their features.
A decision tree is suitable for imbalanced data sets because of the splitting rules that look at
the class variable used in the creation of the tress. However, each data set needs a different
analysis depending on its size and features.
