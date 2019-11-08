import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import argparse


parser = argparse.ArgumentParser(description='input file path')
parser.add_argument('--input_path', type = str, required = True,
                   help='this file will be devided to train, dev, and test datasets')

args = parser.parse_args()


le = LabelEncoder()

df = pd.read_csv(str(args.input_path) + "/train.csv")

df_bert = pd.DataFrame({'id': df['id'],
                'intent': df["intent"],#le.fit_transform(df["intent"]),
                'alpha':['a']* df.shape[0],
                'text': df['text'].replace(r'\n','',regex=True)})

df_bert_train, df_bert_dev = train_test_split(df_bert, test_size=0.01)
#df_bert_train, df_bert_dev = train_test_split(df_bert_train, test_size=0.01)

df_test = pd.read_csv(str(args.input_path) + "/test.csv")
df_bert_test = pd.DataFrame({'id':df_test['id'],
                'text': df_test['text'].replace(r'\n', '', regex=True)})

df_bert_train.to_csv('data/train.tsv', sep='\t', index = False, header= False)
df_bert_test.to_csv('data/test.tsv', sep='\t', index = False, header= True)
df_bert_dev.to_csv('data/dev.tsv', sep='\t', index = False, header= False)


'''
df_results = pd.read_csv("bert_output/test_results.tsv",sep="\t",header=None)
df_results_csv = pd.DataFrame({'id':df_test['id'],
                               'predicted-intent':df_results.idxmax(axis=1),
                               #'encoded_real_inent' : le.transform(df_test["intent"]),
                               'real_intent' : df_test["intent"],
                               'decoded_predicted_intent' : le.inverse_transform(df_results.idxmax(axis=1))})
df_results_csv.to_csv('data/result.csv',sep=",",index=None)
#print(f'{df_results_csv["id"]} : {df_test["intent"]}= {le.inverse_transform(df_results_csv["intent"])}')
        # ' = {le.inverse_transform(i["intent"])}')

''' 
# writing into .csv

