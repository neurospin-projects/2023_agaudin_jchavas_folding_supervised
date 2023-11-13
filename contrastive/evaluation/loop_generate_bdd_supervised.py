import os
import pandas as pd
from datetime import datetime

from contrastive.utils.models_database import *


folders_output = "/neurospin/dico/data/deep_folding/history/2023-09_joel-flanker/Output"
save_path = "/neurospin/dico/data/deep_folding/history/2023-09_joel-flanker/Output/summary"

# Makes the loop when there is only one output folder containing all models in subdirectories

bdd_name_list = ["2023-09-21_SFintermediate", "2023-09-22_internal_parietal",\
                 "2023-09-25_intraparietal", "2023-09-26_central",\
                 "2023-09-26_fronto-parietal_medial_face", "2023-09-26_inferior_temporal",\
                 "2023-09-27_STs_br", "2023-09-28_precentral"]

# construct the database

for bdd_name in bdd_name_list:

    folders = [f"{folders_output}/{bdd_name}"]

    bdd = []
    visited = []

    generate_bdd_models(folders, bdd, visited, dataset="flanker_class", supervised=True, verbose=False)

    bdd = pd.DataFrame(bdd)
    print("Number of models:", bdd.shape[0])
    if bdd.empty:
        raise ValueError(
            "Empty dataframe => no model selected: "
            "you should check the 'folders' parameter.")

    for col in bdd.columns:
        print(col, bdd[col][0])

    # remove useless columns
    bdd = post_process_bdd_models(bdd, hard_remove=[], git_branch=False)
    # the hard remove are the ones containing [] char in their fields.
    # They are (for now) patch_size, partition, numpy_all


    # save the database

    bdd.to_csv(os.path.join(save_path, f"bdd_{bdd_name}.csv"), index=True)


    # write the little readme
    with open(os.path.join(save_path, f"README_{bdd_name}.txt"), 'w') as file:
        file.write("Contient les paramètres de tous les modèles d'intérêt "
                "(dossiers précisés en-dessous). "
                "La base est faite en sorte que seuls les paramètres "
                "qui changent entre les modèles soient enregistrés.\n")
        file.write("\n")
        file.write("Généré avec contrastive/evaluation/generate_bdd_supervised.py le " +
                datetime.now().strftime('%d/%m/%Y à %H:%M') + '.\n')
        file.write("\n")
        file.write("Dossiers utilisés : [")
        for folder in folders:
            file.write(folder)
            if folder == folders[-1]:
                file.write(']')
            else:
                file.write(',\n')
