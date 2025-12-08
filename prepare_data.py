import pandas as pd
import numpy as np


df = pd.read_csv('UCI_Credit_Card.csv')


df['Current_Utilization_Rate'] = df['BILL_AMT1'] / df['LIMIT_BAL']
df['Current_Utilization_Rate'] = np.clip(df['Current_Utilization_Rate'], 0, 1.5)


df['Minimum_Payment_Flag'] = (df['PAY_0'] >= 0).astype(int)


df.rename(columns={'PAY_0': 'Days_Since_Last_Payment_Proxy'}, inplace=True)


df.rename(columns={'default.payment.next.month': 'Target_Delinquent_Next_Month'}, inplace=True)


training_df = df[['ID',
                  'Current_Utilization_Rate',
                  'Minimum_Payment_Flag',
                  'Days_Since_Last_Payment_Proxy',
                  'Target_Delinquent_Next_Month']].copy()

training_df.to_csv('training_data.csv', index=False)