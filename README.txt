DETAILED DESCRIPTIONS OF DATA FILES
====================================

Here are brief descriptions of the data and metadata.
For easy-loading, it is recommended to use pandas, e.g., 

import pandas as pd
df = pd.read_csv(filename, sep="\t")

===DATA DESCRIPTION===

The provided dataset splits (train and test) are in the form of TSV (tab-separated values). 
Altogether, they contain 109,104 records by 2,192 users on 2,640 items. 

The dataset is divided into train and test split based on timestamps: the first 80% of the records constitute the train set, and the rest are in the test split.

The first row contains the header names (user_id, item_id, rating, timestamp).
Each row after the first row represents a user-item interaction.

Note that these splits have been subject through a series of data filtering, from the files available at: https://github.com/RUCAIBox/RecSysDatasets/tree/master.

===METADATA DESCRIPTION===
The provided metadata file is in the form of TSV. 
The first row contains the header names. 
Please refer to the original dataset repository and research paper for more information on the columns: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

===RUN DESCRIPTION===
Just execute the main function to run the evaluation.
FOr Hit Rate and Mrr execute hits.py