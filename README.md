### Active Learning Pipeline for system log labelling

To run: pre-process dataset into Pandas dataframe that contains the following lines: **timestamp** 	**event** 	**machine** 	**label**

Call tokenize method in tokenize.py to train the tokenizer, then pass the tokenizer and the pre-processed dataset into train.py train method. 

Example usage can be found in `alert/alert_wilson.ipynb`
