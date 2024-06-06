## Business Understanding

#### Business Problem
The telecom industry faces a high rate of customer churn, customer churn refers to the phenomenon where customers stop using the company’s services and switch to competitors. High churn rates negatively impact the company’s revenue, market share, and overall profitability. The primary business problem Syriatel faces is understanding the underlying factors driving customer churn and developing effective strategies to retain customers

#### Objectives
1. Develop machine learning models to predict customer churn using customer feature data.
2. Compare the performance of different models to determine the most accurate one for prediction.
3. Identify specific features significantly impacting customer churn rates.
4. Provide actionable recommendations based on analysis to help Seriatel reduce churn rates and improve customer retention.

### Data Understanding
The project uses historical customer data, including demographic and transactional information from Seriatel, to build a predictive model that classifies customers as churned or non-churned. The dataset comprises 3333 rows and 21 columns, with the company based in California, USA. The column titles are:

- **state**: Customer's state of residence.
- **area code**: Area code of the customer's phone number.
- **international plan**: Binary variable indicating if the customer has an international calling plan (1) or not (0).
- **voice mail plan**: Binary variable indicating if the customer has a voicemail plan (1) or not (0).
- **number vmail messages**: Number of voicemail messages the customer has.
- **total day minutes**: Total daytime usage minutes.
- **total day calls**: Total daytime calls made or received.
- **total day charge**: Total daytime usage charges.
- **total eve minutes**: Total evening usage minutes.
- **total eve calls**: Total evening calls made or received.
- **total eve charge**: Total evening usage charges.
- **total night minutes**: Total night usage minutes.
- **total night calls**: Total night calls made or received.
- **total night charge**: Total night usage charges.
- **total intl minutes**: Total international call minutes.
- **total intl calls**: Total international calls made.
- **total intl charge**: Total international call charges.
- **customer service calls**: Number of customer service calls made by the customer.
- **churn**: Binary variable indicating if the customer has churned (1) or not (0).
- **total_calls**: Total calls made or received across all periods (day, evening, night).
- **total_charge**: Total charges incurred across all usage periods.


### Data Processing for Customer Churn Analysis

#### Introduction
In conducting a customer churn analysis, it is crucial to outline the steps and methodologies used to process the data. This statement provides a comprehensive overview of the data processing techniques employed to ensure accurate and meaningful insights into customer churn patterns.

#### Data Preprocessing
Preprocessing is essential to clean and prepare the data for analysis. The following steps were undertaken:
- **Data Cleaning**: Removal of duplicates, handling of missing values, and correction of inconsistencies, identifying outliers.

- **Data Integration**: Merging data from different sources to create a unified dataset.
- **Data Transformation**: Standardizing data formats, converting categorical variables into numerical formats (e.g., one-hot encoding), and normalizing numerical variables.
- **Feature Engineering**: Creating new features that can provide additional insights into churn patterns, such as customer tenure, average response time, and frequency of service use.

#### Data Analysis Techniques
Various analytical techniques were employed to identify and understand customer churn:
- **Descriptive Statistics**: Summarizing the central tendencies, dispersions, and distributions of the data.
- **Correlation Analysis**: Identifying relationships between different variables and their influence on churn.

![alt text](Images/correlation_matrix.PNG)

- **Predictive Modeling**: Using machine learning models such as logistic regression, decision trees, and random forests to predict the likelihood of churn.
- **Clustering**: Grouping customers with similar behaviors and attributes to identify distinct segments with varying churn risks.
- **Survival Analysis**: Estimating the time until churn for different customer segments.

#### Model Validation and Evaluation
To ensure the reliability and accuracy of the predictive models, the following steps were taken:
- **Cross-Validation**: Splitting the data into training and test sets to validate the model's performance.
- **Performance Metrics**: Using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC to evaluate model performance.
- **Baseline Logistic Regression**
The logistic regression model shows significantly better performance for class 0 across all metrics (precision, recall, F1-score) compared to class 1. The overall accuracy of the model is 86.1%, indicating a relatively high rate of correct predictions on the test data.

- **Hyperparameter Tuning**: Optimizing model parameters to improve predictive accuracy and robustness.

