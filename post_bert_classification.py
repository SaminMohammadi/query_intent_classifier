import pandas as pd
from pandas import DataFrame
import argparse

parser = argparse.ArgumentParser("input arguments")
parser.add_argument("--dataset_type", required = True, type=str,
                     help="dataset type such as: 'bank', 'atis', 'assurance', ...")
args = parser.parse_args()




df_test = pd.read_csv("data/test.tsv",sep="\t")

df_results = pd.read_csv("bert_output_boursorama/test_results.tsv",sep="\t",header=None)

df_results_csv = pd.DataFrame({'id':df_test.iloc[:,0],
                               #'real_intent': df_test.iloc[0:, 1],
                               'predicted_intent1':df_results.idxmax(axis=1),
                               'predicted_intent2':df_results.T.apply(lambda x: x.nlargest(2).idxmin()),
                               'query': df_test.iloc[0:, 1]})

if args.dataset_type=="bank":
    df_results_csv["predicted_intent1"].replace(0,'Informations', inplace=True)
    df_results_csv["predicted_intent1"].replace(1,'Qualitatif', inplace=True)
    df_results_csv["predicted_intent1"].replace(2,'Renseignements', inplace=True)
    df_results_csv["predicted_intent1"].replace(3,'Simulation/Comparaison', inplace=True)
    df_results_csv["predicted_intent1"].replace(4,'Spécificités Produit', inplace=True)

    df_results_csv["predicted_intent2"].replace(0,'Informations', inplace=True)
    df_results_csv["predicted_intent2"].replace(1,'Qualitatif', inplace=True)
    df_results_csv["predicted_intent2"].replace(2,'Renseignements', inplace=True)
    df_results_csv["predicted_intent2"].replace(3,'Simulation/Comparaison', inplace=True)
    df_results_csv["predicted_intent2"].replace(4,'Spécificités Produit', inplace=True)

if args.dataset_type=="assurance":
    df_results_csv["predicted_intent1"].replace(0,'Comparateur / Simulation', inplace=True)
    df_results_csv["predicted_intent1"].replace(1,'Information', inplace=True)
    df_results_csv["predicted_intent1"].replace(2,'Qualitatif', inplace=True)
    df_results_csv["predicted_intent1"].replace(3,'Renseignements', inplace=True)
    df_results_csv["predicted_intent1"].replace(4,'Spécificité Produit', inplace=True)

    df_results_csv["predicted_intent2"].replace(0,'Comparateur / Simulation', inplace=True)
    df_results_csv["predicted_intent2"].replace(1,'Information', inplace=True)
    df_results_csv["predicted_intent2"].replace(2,'Qualitatif', inplace=True)
    df_results_csv["predicted_intent2"].replace(3,'Renseignements', inplace=True)
    df_results_csv["predicted_intent2"].replace(4,'Spécificité Produit', inplace=True)

df_results_csv.to_csv('bert_output_boursorama/result_'+args.dataset_type+'.csv',sep=",",index=None)



