
def explainer_to_htx(explainer_dict):
    '''
    '''
    return_dict = {}
    for key, section in explainer_dict.items():

        htx_dict = {}
        htx_dict['doc_id'] = section['metadata']['doc_id']
        htx_dict['actual'] = section['metadata']['actual']
        ### Get top contirbuting model
        model_weighting_list = []
        for sub_section in section['explainer_dict'].values():
            model_weighting_list.append(sub_section['model_weighting'])
        
        percentages = {}
        for sub_key, sub_section in section['explainer_dict'].items():
            percentages[sub_key] = round((sub_section['model_weighting'] / sum(model_weighting_list)),2)

        ### Get top contributing network using model_weighting_list
        network_contributions = {}
        for sub_section in section['explainer_dict'].values():
            network_type = sub_section['network_type']
            if network_type not in network_contributions:
                network_contributions[network_type] = 0
            network_contributions[network_type] += sub_section['model_weighting']
        max_network = max(network_contributions, key=network_contributions.get)

    
        ### Get top information type
        information_contributions = {}
        for sub_section in section['explainer_dict'].values():
            information_type = sub_section['information_type']
            if information_type not in information_contributions:
                information_contributions[information_type] = 0
            information_contributions[information_type] += sub_section['model_weighting']
        max_information = max(information_contributions, key=information_contributions.get)


        ### Get top reliability factors
        reliability_factors = {}
        for sub_section in section['explainer_dict'].values():
            for factor, value in sub_section.get('reliability_factors', {}).items():
                if factor not in reliability_factors:
                    reliability_factors[factor] = value
                else:
                    reliability_factors[factor] = max(reliability_factors[factor], value)


        max_reliability_value = max(reliability_factors.values())                                           # This is to get a list of the max ones if there are multiple
        max_reliability_values = {}
        for factor, value in reliability_factors.items():
            if value == max_reliability_value:
                max_reliability_values[factor] = value


        # If equal contribution of all models - return all 
        if len(set(percentages.values())) == 1:
            htx_dict['htx_model'] = percentages

        else:      
            htx_dict['htx_model'] = {max(percentages, key=percentages.get): percentages[max(percentages, key=percentages.get)]}


        # If equal contribution of all networks - return all
        if len(set(network_contributions.values())) == 1:
            htx_dict['htx_network'] = network_contributions

        else:
            htx_dict['htx_network'] = {max_network:round(network_contributions[max_network]/sum(model_weighting_list),2)}


        # If equal contribution of all information types - return all
        if len(set(information_contributions.values())) == 1:
            htx_dict['htx_information'] = information_contributions

        else:
            htx_dict['htx_information'] = {max_information:round(information_contributions[max_information]/sum(model_weighting_list),2)}


        # note this does not need the if equal part, since return the max values everytime - regardless of how many keys are included. 
        htx_dict['htx_reliability'] = {max(reliability_factors, key=reliability_factors.get): max_reliability_values}

        return_dict[key] = htx_dict

    return(return_dict)



