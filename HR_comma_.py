# import packages
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Extract the data
df = pd.read_csv("C:\\To Read\\Data_sets\\HR_comma_sep.csv")

# Finding impact on employee retention
df.groupby(df['left']).mean()

# Bar chart showing impact of employee salaries on retention
pd.crosstab(df['Department'],df['left']).plot(kind='bar')

# Training Input and Output
input_df1 = df.drop(columns=['last_evaluation', 'number_project','time_spend_company','Work_accident','left','Department'])
input_df = pd.get_dummies(input_df1,dtype=int)
output_df = df['left']
input_df_train, input_df_test, output_df_train, output_df_test = train_test_split(input_df, output_df, train_size=0.8)

# Model and its prediction
model = LogisticRegression()
model.fit(input_df_train, output_df_train)
prediction = model.predict(input_df_test)

# Score
model.score(input_df_train,output_df_train)