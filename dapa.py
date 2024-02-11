import os
import pandas as pd
import time
import tqdm

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


    ### Aggregator ###
    df_list = []
    explainer_list = []

    ### Loop over ids ###
    for _, row in tqdm.tqdm(inputdata_df.iterrows()):        
        row_metadata = metadata_df[metadata_df['doc_id'] == row['doc_id']]                                  # Filter the metadata to the specific doc_id, actual, publisher, word_count, document_count, publisher_type, item_count, item_per_user, document_age
        probability_dict, explainer_dict = functions_aggregator.aggregator(row.to_dict(), row_metadata.to_dict(orient='records')[0], model_lookup_dict, bm_performance_weightings)
        

        ### Merge all probability dictionaries ###
        if probability_dict is not None:
            df_list.append(probability_dict)


        ### Save all Explainability files ###
        for key, value in explainer_dict.items():
            doc_id = value['metadata']['doc_id']
            explainer_path = os.path.join(explainer_dir, f'explainer-{doc_id}-{key}.json')
            tools.save_json(explainer_path, value)
            print(explainer_path)


    ### Save Probability Dataframe ###
    df1 = pd.DataFrame(df_list)
    tools.df_to_csv(outputdata_path, df1)

    
    ### Print output ###
    print("\n### df1 ###")
    print(df1)
    


    ### Print time taken ###
    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken:.4f} seconds")




if __name__ == '__main__':
    main()

