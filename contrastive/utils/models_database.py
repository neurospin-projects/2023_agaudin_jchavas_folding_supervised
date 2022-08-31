import os
import pandas as pd
import json
import yaml


# functions to create a database containing all the models
# These functions are used in the generate_bdd notebook


def get_subdirs(directory):
    sub_dirs = os.listdir(directory)
    sub_dirs = [os.path.join(directory, name) for name in sub_dirs]
    sub_dirs = [path for path in sub_dirs if os.path.isdir(path)] # remove files
    return sub_dirs


def process_model(model_path, dataset='cingulate_ACCpatterns', verbose=True):
    # generate a dictionnary with the model's parameters and performances
    model_dict = {}
    model_dict['model_path'] = model_path
    # read performances
    with open(model_path + f"/{dataset}_embeddings/values.json", 'r') as file:
        values = json.load(file)
        decomposed_values = {'auc': values['cross_val_auc'][0],
                            'auc_std': values['cross_val_auc'][1],
                            'accuracy': values['cross_val_total_accuracy'][0],
                            'accuracy_std': values['cross_val_total_accuracy'][1]}
        model_dict.update(decomposed_values)
    # read parameters
    with open(model_path+'/partial_config.yaml', 'r') as file2:
        partial_config = yaml.load(file2, Loader=yaml.FullLoader)
        model_dict.update(partial_config)
    
    return model_dict


def generate_bdd_models(folders, bdd_models, visited, dataset='cingulate_ACCpatterns', verbose=True):
    # fill the dictionnary bdd_models with the parameters and performances of all the bdd models
    # depth first exploration of folders to treat all the models in it
    
    if verbose:
        print("Start", len(folders), len(bdd_models))

    while folders != []:
        # remove folders already treated
        folders = [folder for folder in folders if folder not in visited]
        
        # condition as folders can be emptied by the previous line
        if folders != []:
            dir_path = folders.pop()
            visited.append(dir_path)
            
            # checks if directory
            if os.path.isdir(dir_path):
                # check if directory associated to a model
                if os.path.exists(dir_path+'/.hydra/config.yaml'):
                    print("Treating", dir_path)
                    # check if values and parameters computed for the model
                    if os.path.exists(dir_path + f"/{dataset}_embeddings/values.json"):
                        model_dict = process_model(dir_path)
                        bdd_models.append(model_dict)
                        if verbose:
                            print("End model", len(folders), len(bdd_models))

                    else:
                        print(f"Model does not have embeddings and their evaluation OR \
they are done with another database than {dataset}")

                else:
                    print(f"{dir_path} not associated to a model. Continue")
                    new_dirs = get_subdirs(dir_path)
                    folders.extend(new_dirs)
                    # remove folders already treated
                    folders = [folder for folder in folders if folder not in visited]
                    if verbose:
                        print("End recursive", len(folders), len(bdd_models))
                    
                    generate_bdd_models(folders, bdd_models, visited, dataset=dataset, verbose=verbose)
            
            else:
                print(f"{dir_path} is a file. Continue.")
                if verbose:
                    print("End file", len(bdd_models))


def post_process_bdd_models(bdd_models, hard_remove=[], git_branch=False):
    # specify dataset if not done
    if "dataset_name" in bdd_models.columns:
        bdd_models.numpy_all.fillna(value="osef", inplace=True)
        bdd_models.dataset_name.fillna(value="cingulate_HCP_half_1", inplace=True)
        bdd_models.loc[bdd_models.numpy_all.str.contains('1mm'), 'dataset_name'] = "cingulate_HCP_1mm"
    
    # hard_remove contains columns you want to remove by hand
    bdd_models = bdd_models.drop(columns=hard_remove)

    # remove duplicates (normally not needed)
    bdd_models.drop_duplicates(inplace=True, ignore_index=True)

    # deal with '[' and ']'

    # specify git branch
    if git_branch:
        bdd_models['git_branch'] = ['Run_03_aymeric' for i in range(bdd_models.shape[0])]
        bdd_models.loc[bdd_models.backbone_name.isna(), 'git_branch'] = 'Run_43_joel'
        bdd_models.loc[bdd_models.backbone_name == 'pointnet', 'git_branch'] = 'pointnet'


    # remove columns where the values never change
    remove = []
    for col in bdd_models.columns:
        col_values = bdd_models[col].dropna().unique()
        if len(col_values) <= 1:
            remove.append(col)
    bdd_models = bdd_models.drop(columns=remove)

    # sort by model_path
    bdd_models.sort_values(by="model_path", axis=0, inplace=True, ignore_index=True)


    return bdd_models