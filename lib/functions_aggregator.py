import numpy as np


def aggregator(df_row, metadata_row, model_lookup_dict, bm_performance_weightings):
    '''
    This function is run for each row of the dataframe.
    input
    - df_row : a row of the dataframe
    - metadata_row : a row of the row_metadata
    - model_lookup_dict : the config - model, infomation, reliabilty factors, model_dir, filename
    - bm_performance_weightings : the weightings of the base models
    output
    - probability_dict : the base model probabilities and the aggregation probabilities
    - explainer_dict : dictionary for HTX - DAPA and BMAcc
    '''
    ### Extract variables from input data ###
    base_models1 = model_lookup_dict.keys()
    dataset = df_row["dataset"]


    ### Run Aggreagtor ### 
    dapa_explainer_dict = dapa_function(metadata_row, df_row, model_lookup_dict, base_models1)
    bmacc_explainer_dict = bmAcc_function(metadata_row, df_row, base_models1, bm_performance_weightings[dataset])
    max_value = max_function(df_row, base_models1)
    average_value = av_function(df_row, base_models1)
    

    ### Probability dictionary ###
    probability_dict = {}
    probability_dict['doc_id'] = df_row['doc_id']
    probability_dict['actual'] = metadata_row['actual']
    probability_dict['DAPA'] = dapa_explainer_dict["probability"]
    probability_dict['BMAcc'] = bmacc_explainer_dict["probability"]
    probability_dict['Max'] = max_value
    probability_dict['Av'] = average_value


    ### Combine the base models and the aggregation probabilities ###
    base_models = get_model_probs_dict(df_row, base_models1)            # Dictionary of all base models
    for model in base_models.keys():
        probability_dict[model] = round(base_models[model], 2)


    ### Explainer dictionary ###
    explainer_dict = {'DAPA': dapa_explainer_dict, 'BMAcc': bmacc_explainer_dict}

    return probability_dict, explainer_dict



def bmAcc_function(metadata_row, df_row, base_models, dataset_weights):
    '''
    Aggregator - bmAcc of all base models
    weight probabilities of training model performance 
    '''
    prob_dict_temp = get_model_probs_dict(df_row, base_models)
    explainer_dict = {}
    prob_list_temp = []
    weight_list_temp = []
    for model in prob_dict_temp.keys():
        bmprob = prob_dict_temp[model]
        bmweight = dataset_weights[model]
        
        explainer_dict_temp = {}
        explainer_dict_temp['bm_probability'] = round(bmprob,2)
        explainer_dict_temp['model_weighting'] = round(bmweight,2)
        explainer_dict_temp['weighted_probability'] = round(bmweight*bmprob,2)
        explainer_dict[model] = explainer_dict_temp

        prob_list_temp.append(bmprob)
        weight_list_temp.append(bmweight)

    explainer_dict_full = {}
    explainer_dict_full["metadata"] = {'doc_id': df_row['doc_id'], 'actual': metadata_row['actual']}
    explainer_dict_full["probability"] = round(np.average(prob_list_temp, weights=weight_list_temp), 2)                       # Weighted average score
    explainer_dict_full["explainer_dict"] = explainer_dict

    return explainer_dict_full



def av_function(df_row, base_models):
    '''
    Aggregator - average of all base models
    '''
    prob_list_temp = get_model_probs_list(df_row, base_models)
    aggregation_av = round(np.average(prob_list_temp),2)
    
    return aggregation_av



def max_function(df_row, base_models):
    '''
    Aggregator - max of all base models
    '''
    prob_list_temp = get_model_probs_list(df_row, base_models)
    updated_values = [abs(original - 0.5) for original in prob_list_temp]                                   # Subtract the common value from each element in the list
    paired_values = {original: updated for original, updated in zip(prob_list_temp, updated_values)}        # Create a dictionary with original values as keys and updated values as values
    largest_updated_value = max(updated_values)                                                             # Find the largest of the updated values
    corresponding_original_value = [original for original, updated in paired_values.items() if updated == largest_updated_value][0]

    aggregation_max = round(corresponding_original_value,2)

    return aggregation_max



def dapa_function(metadata_row, df_row, model_lookup_dict, base_models):
    '''
    Aggregator - Dynamic adaptive probability aggregation (DAPA) 
    weight probabilities based on reliability factors
    '''
    ### Get raw values of the reliability factors ###
    dataset = df_row['dataset']  
    doc_id = df_row['doc_id']
    actual = metadata_row['actual']
    publisher = metadata_row['publisher']
    word_count = int(metadata_row['word_count'])
    publisher_type = metadata_row['publisher_type']
    document_count = int(metadata_row['document_count'])
    item_count = int(metadata_row['item_count'])
    item_per_user = int(metadata_row['item_per_user'])
    document_age = int(metadata_row['document_age'])

    ### The relaibilty scores ### 
    dictin = {"politifact": {
                'word_count_values': {25: 0, 100:0.4, 300:0.6, 600:0.8, 800:0.6},
                'publisher_type_values': {'new': 0.1, 'exist':1},
                'document_count_values': {1: 0.1, 3:0.4, 8:0.5, 9:1},
                'item_count_values': {1: 0.2, 10:0.4, 50:0.7, 51:1},
                'item_per_user_values': {1: 0.1, 3:0.2, 8:0.5, 9:1},
                'document_age_values': {2: 0.01, 24:0.1, 24*7:0.4, 24*7*2:1}
                }}



    ### Get weights for each reliability factor ###
    word_count_weight = calculate_weight_int(word_count, dictin,'word_count_values', dataset)
    publisher_type_weight = calculate_weight_str(publisher_type, dictin,'publisher_type_values', dataset)
    document_count_weight = calculate_weight_int(document_count, dictin,'document_count_values', dataset)
    item_count_weight = calculate_weight_int(item_count, dictin,'item_count_values', dataset)
    item_per_user_weight = calculate_weight_int(item_per_user, dictin,'item_per_user_values', dataset)
    document_age_weight = calculate_weight_int(document_age, dictin,'document_age_values', dataset)
    return_weightdict = {'word_count_weight':word_count_weight, 'word_count': word_count,
                         'publisher_type_weight':publisher_type_weight, 'publisher_type': publisher_type,
                         'document_count_weight':document_count_weight, 'document_count': document_count,
                         'item_count_weight':item_count_weight, 'item_count': item_count,
                         'item_per_user_weight':item_per_user_weight, 'item_per_user': item_per_user,
                         'document_age_weight':document_age_weight, 'document_age': document_age                                
                                }


    ### Weight model probabilites using reliability factors ###
    explainer_dict = {}
    prob_list_temp = []
    weight_list_temp = []
    for base_model in df_row.keys():

        if base_model not in base_models:
            continue

        bmprob = df_row[base_model]

        explainer_dict_temp = {}
        explainer_dict_temp['network_type'] = model_lookup_dict[base_model]['network']
        explainer_dict_temp['information_type'] = model_lookup_dict[base_model]['information']

        weights_temp = []
        reliability_factors_dict = {}
        for factor in model_lookup_dict[base_model]['reliability_factors']:
            weight_temp = return_weightdict[factor + '_weight']
            reliability_factors_dict[factor] = weight_temp
            weights_temp.append(weight_temp)
        
        reliabilty_score = np.average(weights_temp)
        
        explainer_dict_temp['bm_probability'] = round(bmprob,2)
        explainer_dict_temp['reliability_score'] = round(reliabilty_score,2)
        explainer_dict_temp['weighted_probability'] = round(reliabilty_score*bmprob,2)
        explainer_dict_temp['reliability_factors'] = reliability_factors_dict

        explainer_dict[base_model] = explainer_dict_temp
        
        prob_list_temp.append(bmprob)
        weight_list_temp.append(reliabilty_score)
    
    explainer_dict_full = {}
    explainer_dict_full["metadata"] = {'doc_id': doc_id, 'actual': actual, 'publisher': publisher}
    explainer_dict_full["probability"] = round(np.average(prob_list_temp, weights=weight_list_temp),2)                       # Weighted average score
    explainer_dict_full["explainer_dict"] = explainer_dict
    explainer_dict_full["weight_details"] = return_weightdict

    return explainer_dict_full

    

def get_model_probs_dict(df_row, base_models):
    '''
    '''
    prob_dict_temp = {}
    for base_model in df_row.keys():
        if base_model in base_models:
            prob_dict_temp[base_model] = df_row[base_model]
    return prob_dict_temp



def get_model_probs_list(df_row, base_models):
    '''
    '''
    prob_list_temp = []
    for base_model in df_row.keys():
        if base_model in base_models:
            prob_list_temp.append(df_row[base_model])
    return prob_list_temp



def calculate_weight_int(valuein, dictin, factorin, datasetin):
    '''
    '''
    for threshold, weight in sorted(dictin[datasetin][factorin].items()):
        if valuein <= threshold:
            return weight

    ### Else return max ###
    return dictin[datasetin][factorin][max(dictin[datasetin][factorin])]



def calculate_weight_str(valuein, dictin, factorin, datasetin):
    '''
    '''
    for threshold, weight in dictin[datasetin][factorin].items():
        if valuein == threshold:
            return weight
    
    ### Else ###
    print('ERROR - calculate_weight_str - value not found in dictin')
    return 0
