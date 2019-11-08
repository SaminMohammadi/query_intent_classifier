# query_intent_classifier
This project aims to identify the intention of user' query. The base of this model is the pre-defined google-bert and we fune-tuned this model with our data set to classify the intention of queries.

+ download google bert from  https://github.com/google-research/bert
+ download one of the pre-trained versions of bert
+ create data folder and put yout data in 
+ call pre_bert_classification
+ call run_classifier.py in two next steps: 
##### run bert as classifier (train and test)
python3 run_classifier.py --task_name=bank  --do_train=true --do_eval=true --do_predict=true --data_dir=./data/ --vocab_file=./multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=./multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=./multi_cased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=100 --train_batch_size=8 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=./bert_output/ --do_lower_case=False
##### run bert as predictor only: (without trainning)
python3 run_classifier.py --task_name=bank  --do_train=false --do_eval=false --do_predict=true --data_dir=./data/ --vocab_file=./multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=./multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=./bert_output/model.ckpt-374 --max_seq_length=100 --train_batch_size=8 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=./bert_output/ --do_lower_case=False
