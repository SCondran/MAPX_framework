# Dynamic Adaptive Probability Aggregation (DAPA) #
DAPA is a dynamic aggregation technique which assigns different weights to each of the base models. However, unlike traditional dynamic aggregation techniques which weight the contribution of each base model differently based on the performance during the training process, DAPA adaptively adjusts the weights for each base model based on the reliability score associated with the instance of the document under consideration. 



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
  - Factor reliability information for each document



## Output Directory ##
Output files of the dapa code
- Aggregator output
  - Probability of falsehood for each document (0 True / 1 False)
  - For a number of different models

- Explainer Data
  - Hierarchical Tiered explanation for each document



## Temp Explainer Directory ##
Contains temporary files used in the process of generating the Explainer Data
