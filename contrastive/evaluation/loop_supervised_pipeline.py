from contrastive.evaluation.supervised_pipeline import *

base_path = "/neurospin/dico/data/deep_folding/history/2023-09_joel-flanker/Output"
list_folders = ["2023-09-26_lobule_parietal_sup"]

for folder in list_folders:
    dataset_base = '_'.join(folder.split('_')[1:])
    pipeline(f"{base_path}/{folder}",
            datasets=[f"{dataset_base}_HCP_stratified_extreme_Flanker_left",
                      f"{dataset_base}_HCP_stratified_extreme_Flanker_right"],
            label='Flanker_AgeAdj_class',
            short_name='flanker_class',
            overwrite=True,
            use_best_model=True,
            save_outputs=True)