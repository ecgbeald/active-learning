### Active Learning Pipeline for system log labelling

To run: pre-process dataset into Pandas dataframe that contains the following lines: **timestamp** 	**event** 	**machine** 	**label**

Call tokenize method in tokenize.py to train the tokenizer, then pass the tokenizer and the pre-processed dataset into train.py train method. 

Example usage can be found in `alert/alert_wilson.ipynb`

The overall workflow of our active learning framework consists of the following steps:
- Preprocess the dataset and construct windows of event sequences (document).
- Pre-train a BERT model on the entire unlabelled corpus using Masked Language Modelling (MLM).
- Transfer the model weights to a BERT model with a classification head, treat this as the base model.
- Fine-tune the base model as a binary classifier on a minimal set of randomly sampled labelled examples from each class, the remaining data are initialised as the unlabelled pool.
- Apply the active learning sampling algorithm to select unlabelled logs from the pool.
- Fine-tune the base model using all currently labelled logs.
- Repeat until the termination criteria are met.
