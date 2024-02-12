# Dynamic Adaptive Probability Aggregation (DAPA) #
DAPA is a dynamic aggregation technique which assigns different weights to each of the base models. However, unlike traditional dynamic aggregation techniques which weight the contribution of each base model differently based on the performance during the training process, DAPA adaptively adjusts the weights for each base model based on the reliability score associated with the instance of the document under consideration. 

# Hierarchical Tiered eXplanation (HTX) #
HTX method provides a hierarchical four-tiered explanation on how each model contributes to the final prediction and the impact of reliability factors on the models' performance. Tier 1 identifies the top contributing models; Tier 2 identifies the type of network (\emph{i.e.} content $\mathcal{S}_n$ or context $\mathcal{S}_x$) associated with the top contributing model; Tier 3 then identifies the top contributing information for the network identified in Tier 2; and finally Tier 4 identifies the reliability factors and their corresponding scores of factors in Tier 3. 


## Usage ##
- Code was developed using python 3 on Ubuntu linux 

1. Install python requirements
```
pip3 install -r requirements.txt
```


2. Run the Python script:
```
python3 dapa.py
```


## Data Directory ##
2 demo files containing data for a number of documents
- dataset data 
  - Merged base model probabilities for each document

- metadata 
  - Reliability factors information for each document

- Reliability factor - word_count
  - This is where the lookup table reliability scores are derived
  - Each reliability factor has its own figure to derive the scores. These ones are calculated for each dataset using the entire database. 


## Output Directory ##
Output files of the dapa code
- Aggregator output
  - Probability of falsehood for each document (0 True / 1 False)
  - For a number of different models

- Explainer Data
  - Hierarchical Tiered eXplanation for each document

