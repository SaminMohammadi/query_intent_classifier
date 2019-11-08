######################################
# Converting input datasets including train, dev, and test to
# a format acceptable for the BERT algorithm.
# ####################################
#  
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import argparse
import csv

parser = argparse.ArgumentParser(description='input file name')
parser.add_argument('--input_train_file', type = str, required = True,
                   help='file & path, This file will be devided to train, dev, and probably test datasets depends on input dataset')
parser.add_argument('--input_test_file', type = str, required = False,
                   help='Test dataset')
parser.add_argument('--dataset_type', type = str, required = True,
                   help="Dataset type such as: 'atis', 'bank', 'assurance'")
args = parser.parse_args()


with open(str(args.input_train_file)) as file_read:
    data = csv.reader(file_read)
    headers = next(data) 
    with open("./data/train.csv","w") as file_write:
        csv_file = csv.writer(file_write)
        csv_file.writerow(['id', 'intent', 'text'])
        for i, item in enumerate(data, start=1):
            csv_file.writerow([str(i),item[0], item[1]])
if args.input_test_file:
     with open(str(args.input_test_file)) as file_read:
        data = csv.reader(file_read)
        headers = next(data) 
        with open("./data/test.csv","w") as file_write:
                csv_file = csv.writer(file_write)
                csv_file.writerow(['id', 'intent', 'text'])
                for i, item in enumerate(data, start=1):
                        csv_file.writerow([str(i),item[0], item[1]])


df = pd.read_csv("./data/train.csv")

df_bert = pd.DataFrame({'id': df['id'],
                'intent': df['intent'], 
                'alpha':['a']* df.shape[0],
                'text': df['text'].replace(r'\n','',regex=True)})

df_bert_train, df_bert_dev = train_test_split(df_bert, test_size=0.01)

if args.dataset_type in ('bank', 'assurance'):
        df_bert_train, df_test = train_test_split(df_bert_train, test_size=0.2)
        df_test.to_csv('data/test.csv', sep='\t', index = False, header= False)
elif args.dataset_type == "atis":
        df_test = pd.read_csv("./data/test.csv","w")


df_bert_test = pd.DataFrame({'id':df_test['id'],
                'text': df_test['text'].replace(r'\n', '', regex=True)})

# datasets should be saved in 'data' folder in Bert directory.
df_bert_train.to_csv('data/train.tsv', sep='\t', index = False, header= False)
df_bert_test.to_csv('data/test.tsv', sep='\t', index = False, header= True)
df_bert_dev.to_csv('data/dev.tsv', sep='\t', index = False, header= False)


