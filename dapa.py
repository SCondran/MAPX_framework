import os
import pandas as pd
import time
import tqdm
import json

from lib import functions_aggregator, functions_HTX, tools


def main():
    '''
    '''
    print("\n\n##########################################################")
    print("#################### DAPA Aggregation ####################")
    print("##########################################################")

    start_time = time.time()

    ### Model Lookup ###
    model_lookup_dict = {
        'PC': {'network':'context', 'information':'publisher_history','reliability_factors':['publisher_type', 'document_count'], 'model_dir': 'pcs', 'filename': 'FAKENEWS.txt'},
        'UC': {'network':'content', 'information':'user_history','reliability_factors':['item_count', 'item_per_user', 'document_age'], 'model_dir': 'ucs', 'filename': 'FAKENEWS.txt'},
        'FF': {'network':'context','information':'word','reliability_factors':['word_count'],'model_dir': 'fakeflow', 'filename': 'fakeflowOutput.csv'},
    }
    

    ### Base model training performances ###
    bm_performance_weightings = { 'politifact': {'FF': 0.73, 'PC': 0.75, 'UC':0.88},
                                'gossipcop':    {'FF': 0.54, 'PC': 0.54, 'UC':0.84},
                                'fakeHealth':   {'FF': 0,    'PC': 0.31, 'UC':0.05}}


    
    ### Data paths ###
    data_dir = 'data'
    explainer_dir = "temp_explainer"
    output_dir = "output"
    inputdata_path = os.path.join(data_dir, 'demo_dataset.csv')
    metadata_path = os.path.join(data_dir, 'demo_metadata.csv')
    outputdata_path = os.path.join(output_dir, 'demo_output_aggregator.csv')
    

    ### Create output directory ###
    tools.make_dir(explainer_dir)
    tools.make_dir(output_dir)


    ### Load data ###
    inputdata_df = tools.df_load_csv(inputdata_path)                                                              # read in demo dataset  - doc_id, actual, dataset, FF, PC, UC
    metadata_df = tools.df_load_csv(metadata_path)                                                                # read in demo meta dataset  - doc_id, publisher, unique_


    ### Run Aggregator ###
    df_list = []
    explainer_list = []
    for _, row in tqdm.tqdm(inputdata_df.iterrows()):        
        row_metadata = metadata_df[metadata_df['doc_id'] == row['doc_id']]                                  # Filter the metadata to the specific doc_id, actual, publisher, word_count, document_count, publisher_type, item_count, item_per_user, document_age
        probability_dict, explainer_dict = functions_aggregator.aggregator(row.to_dict(), row_metadata.to_dict(orient='records')[0], model_lookup_dict, bm_performance_weightings)
        df_list.append(probability_dict)

        ### Merge all explainer dictionaries ###
        htx_output = functions_HTX.explainer_to_htx(explainer_dict)
        explainer_list.append(htx_output)

    

    ### Explainer Data ###
    explainer_output_dict = {}
    aggregator_methods = explainer_list[0].keys()

    for method in aggregator_methods:                   # Create empty lists for each method
        explainer_output_dict[method] = []

    ## Loop over each explainer and add to the output List of dictionaries ##
    for dict_dict in explainer_list:                    # Loop over documents
        doc_id = dict_dict['DAPA']['doc_id']
        for method in aggregator_methods:               # Loop over methods
            explainer_output_dict[method].append(dict_dict[method])

    ## Save each explainer output to a csv ##
    for model, df_new in explainer_output_dict.items():
        print(f"\n\n### Explainer - {model} ###")
        df_exp = pd.DataFrame(df_new)
        print(df_exp)
        outputdata_path = os.path.join(output_dir, f'explainer_{model}.csv')
        tools.df_to_csv(outputdata_path, df_exp)



    ### Aggregator Data ###
    df_agg = pd.DataFrame(df_list)
    tools.df_to_csv(outputdata_path, df_agg)
    print("\n### Aggregator Data ###")
    print(df_agg)
    


    ### Print time taken ###
    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken:.4f} seconds")




if __name__ == '__main__':
    main()

