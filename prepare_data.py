import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('UCI_Credit_Card.csv')

# F1: Current Utilization Rate
df['Current_Utilization_Rate'] = df['BILL_AMT1'] / df['LIMIT_BAL']
df['Current_Utilization_Rate'] = np.clip(df['Current_Utilization_Rate'], 0, 1.5)

# F2: Minimum Payment Flag (Proxy)
# 1 if PAY_0 is 0 or greater (implies stress/minimum payment/late), 0 otherwise.
df['Minimum_Payment_Flag'] = (df['PAY_0'] >= 0).astype(int)

# F3: Days Since Last Payment (Proxy)
df.rename(columns={'PAY_0': 'Days_Since_Last_Payment_Proxy'}, inplace=True)

# Target Variable (Y)
df.rename(columns={'default.payment.next.month': 'Target_Delinquent_Next_Month'}, inplace=True)

# Select the final columns and save
training_df = df[['ID',
                  'Current_Utilization_Rate',
                  'Minimum_Payment_Flag',
                  'Days_Since_Last_Payment_Proxy',
                  'Target_Delinquent_Next_Month']].copy()

training_df.to_csv('training_data.csv', index=False)