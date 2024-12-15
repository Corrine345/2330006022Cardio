# Q1
import pandas as pd
try:
    data = pd.read_csv("FitTrackData.csv")    
    a1 = len(data)
    print(a1)
except FileNotFoundError:
    print("Error: The file 'FitTrackData.csv' does not exist.")

except pd.errors.EmptyDataError:
    print("Error: The file 'FitTrackData.csv' is empty.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Q2

import pandas as pd
try:
    data = pd.read_csv("FitTrackData.csv")
    data.head()
    if 'DeviceModel' not in data.columns:
        raise KeyError("The 'DeviceModel' column does not exist in the data.")
    
    data['DeviceModel'] = data['DeviceModel'].fillna('Unknown')
    a2 = len(data['DeviceModel'].unique())
    print(a2)

except FileNotFoundError:
    print("Error: The file 'FitTrackData.csv' does not exist.")

except pd.errors.EmptyDataError:
    print("Error: The file 'FitTrackData.csv' is empty.")

except KeyError as e:
    print(f"Error: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Q3

import pandas as pd
try:
    data = pd.read_csv("FitTrackData.csv")
    data.head()
    if 'DeviceModel' not in data.columns:
        raise KeyError("The 'DeviceModel' column does not exist in the data.")
    if 'Age' in data.columns:
        data['Age'] = data['Age'].fillna(data['Age'].mean())
    else:
        raise KeyError("The 'Age' column does not exist in the data.")
    a2 = len(data['DeviceModel'].unique())
    print(a2)

except FileNotFoundError:
    print("Error: The file 'FitTrackData.csv' does not exist.")

except pd.errors.EmptyDataError:
    print("Error: The file 'FitTrackData.csv' is empty.")

except KeyError as e:
    print(f"Error: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Q4

import pandas as pd
try:
    data = pd.read_csv("FitTrackData.csv")
    data.head()
    if 'Gender' not in data.columns:
        raise KeyError("The 'Gender' column does not exist in the data.")
    if data['Gender'].isnull().any():
        data['Gender'].fillna('Unknown', inplace=True)
    female_count = len(data[data['Gender'] == 'Female'])
    male_count = len(data[data['Gender'] == 'Male'])
    if male_count == 0:
        raise ZeroDivisionError("All people are female or unknown.")
    a4 = female_count / male_count 
    print(f"Female to Male Ratio: {a4}")

except FileNotFoundError:
    print("Error: The file 'FitTrackData.csv' does not exist.")

except pd.errors.EmptyDataError:
    print("Error: The file 'FitTrackData.csv' is empty.")

except KeyError as e:
    print(f"Error: {e}")

except ZeroDivisionError as e:
    print(f"Error: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")



# Q5

import pandas as pd
try: 
    data = pd.read_csv("FitTrackData.csv")
    if 'EducationYears' not in data.columns:
        raise KeyError("The 'EducationYears' column does not exist in the data.")
    
    if data['EducationYears'].isnull().any():
        data['EducationYears'].fillna(data['EducationYears'].median(), inplace=True)
    education_median = data['EducationYears'].median()
    print(education_median)

except FileNotFoundError:
    print("Error: The file 'FitTrackData.csv' does not exist.")

except pd.errors.EmptyDataError:
    print("Error: The file 'FitTrackData.csv' is empty.")

except KeyError as e:
    print(f"Error: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Q6

import pandas as pd
try:
    data = pd.read_csv("FitTrackData.csv")
    if 'MaritalStatus' not in data.columns:
        raise KeyError("The 'MaritalStatus' column does not exist in the data.")
    if data['MaritalStatus'].isnull().any():
        data['MaritalStatus'].fillna('Unknown', inplace=True)
    not_single_count = (data['MaritalStatus'] != 'Single').sum()
    print(not_single_count)

except FileNotFoundError:
    print("Error: The file 'FitTrackData.csv' does not exist.")

except pd.errors.EmptyDataError:
    print("Error: The file 'FitTrackData.csv' is empty.")

except KeyError as e:
    print(f"Error: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")




# Q7

import pandas as pd
try:
    data = pd.read_csv("FitTrackData.csv")
    if 'UsageFrequency' not in data.columns:
        raise KeyError("The 'UsageFrequency' column does not exist in the data.")
    if data['UsageFrequency'].isnull().any():
        data['UsageFrequency'].fillna(data['UsageFrequency'].mean(), inplace=True)
    avg_usage_frequency = data['UsageFrequency'].mean()
    print(avg_usage_frequency)

except FileNotFoundError:
    print("Error: The file 'FitTrackData.csv' does not exist.")

except pd.errors.EmptyDataError:
    print("Error: The file 'FitTrackData.csv' is empty.")

except KeyError as e:
    print(f"Error: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Q8
import pandas as pd
try:
    data = pd.read_csv("FitTrackData.csv")
    if 'HealthScore' not in data.columns:
        raise KeyError("The 'HealthScore' column does not exist in the data.")
    if data['HealthScore'].isnull().any():
        data['HealthScore'].fillna(3, inplace=True)
    health_score_five_count = (data['HealthScore'] == 5).sum()
    if len(data) == 0:
        raise ZeroDivisionError("The dataset is empty, cannot perform division.")
    health_score_five_percentage = health_score_five_count / len(data) * 100
    print(f"Percentage of users with HealthScore 5: {health_score_five_percentage:.2f}%")

except FileNotFoundError:
    print("Error: The file 'FitTrackData.csv' does not exist.")

except pd.errors.EmptyDataError:
    print("Error: The file 'FitTrackData.csv' is empty.")

except KeyError as e:
    print(f"Error: {e}")

except ZeroDivisionError as e:
    print(f"Error: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Q9

import pandas as pd
try: 
    data = pd.read_csv("FitTrackData.csv")
    if 'AnnualIncome' not in data.columns:
        raise KeyError("The 'AnnualIncome' column does not exist in the data.")
    if data['AnnualIncome'].isnull().any():
        data['AnnualIncome'].fillna(data['AnnualIncome'].median(), inplace=True)
    max_annual_income = data['AnnualIncome'].max()
    print(max_annual_income)

except FileNotFoundError:
    print("Error: The file 'FitTrackData.csv' does not exist.")

except pd.errors.EmptyDataError:
    print("Error: The file 'FitTrackData.csv' is empty.")

except KeyError as e:
    print(f"Error: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Q10

import pandas as pd
try:
    data = pd.read_csv("FitTrackData.csv")
    if 'StepsPerWeek' not in data.columns:
        raise KeyError("The 'StepsPerWeek' column does not exist in the data.")
    if data['StepsPerWeek'].isnull().any():
        data['StepsPerWeek'].fillna(data['StepsPerWeek'].median(), inplace=True)
    steps_over_50000_count = len(data[data['StepsPerWeek'] > 50000])
    print(steps_over_50000_count)

except FileNotFoundError:
    print("Error: The file 'FitTrackData.csv' does not exist.")

except pd.errors.EmptyDataError:
    print("Error: The file 'FitTrackData.csv' is empty.")

except KeyError as e:
    print(f"Error: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")


# Visual Q1

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("FitTrackData.csv")
data.head()
data['DeviceModel'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Distribution of Device model")
plt.ylabel("")  
plt.show()

# Significance test

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway
data = pd.read_csv("FitTrackData.csv")
fitband100_usage_frequency = data[data['DeviceModel'] == 'FitBand 100']['UsageFrequency']
fitband200_usage_frequency = data[data['DeviceModel'] == 'FitBand 200']['UsageFrequency']
fitband300_usage_frequency = data[data['DeviceModel'] == 'FitBand 300']['UsageFrequency']
f_statistic, p_value = f_oneway(fitband100_usage_frequency, fitband200_usage_frequency, fitband300_usage_frequency)
print("F statistic of the effect of the device model on the frequency of use:", f_statistic)
print("The P-value of the effect of the device model on the frequency of use:", p_value)

#  Visual Q2

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("FitTrackData.csv")
health_score = data['HealthScore']
bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
grouped_data, _ = np.histogram(health_score, bins=bins)
plt.bar(range(1, 6), grouped_data)
plt.xlabel('Health Score')
plt.ylabel('Number of customers')
plt.title('Distribution of Health scores')
plt.show()

# Significance test

import pandas as pd
from scipy import stats
data = pd.read_csv("FitTrackData.csv")
health_score = data['HealthScore']
t_stat, p_value = stats.ttest_1samp(health_score, 3)
print(f"t-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
alpha = 0.05  # Significance level
if p_value < alpha:
    print("The mean health score is significantly greater than 3.")
else:
    print("The mean health score is not significantly greater than 3.")


#  Visual Q3

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
data.groupby(['Gender', 'MaritalStatus'])['AnnualIncome'].mean().plot(kind='bar', ax=ax, rot=0)
ax.set_title('Average Annual Income by Marital Status and Gender')
ax.set_xlabel('Marital Status')
ax.set_ylabel('Average Annual Income')
plt.show()

# Significance test

import pandas as pd
import numpy as np
from scipy.stats import f_oneway
data = pd.read_csv("FitTrackData.csv")
single_income = data[data['MaritalStatus'] == 'Single']['AnnualIncome']
married_income = data[data['MaritalStatus'] == 'Married']['AnnualIncome']   
divorced_income = data[data['MaritalStatus'] == 'Divorced']['AnnualIncome']
f_statistic, p_value = f_oneway(single_income, married_income, divorced_income)
print("F statistic of the effect of marital status on annual income:", f_statistic)
print("The P-value of the effect of marital status on annual income:", p_value)