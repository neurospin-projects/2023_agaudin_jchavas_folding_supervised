import pandas as pd
from datetime import datetime

from contrastive.utils.models_database import *


# construct the database

folders = ["/neurospin/dico/agaudin/Runs/09_new_repo/Output/converter_test/"]
bdd = []
visited = []

generate_bdd_models(folders, bdd, visited, supervised=True, verbose=False)

bdd = pd.DataFrame(bdd)
print("Number of models:", bdd.shape[0])
if bdd.empty:
    raise ValueError(
        "Empty dataframe => no model selected: "
        "you should check the 'folders' parameter.")

for col in bdd.columns:
    print(col, bdd[col][0])

# remove useless columns
bdd = post_process_bdd_models(bdd, hard_remove=["partition"], git_branch=False)
# the hard remove are the ones containing [] char in their fields.
# They are (for now) patch_size, partition, numpy_all


# save the database
name = "schiz_strat_bis"
save_path = "/neurospin/dico/agaudin/Runs/09_new_repo/Output/converter_test/"
bdd.to_csv(save_path+f"bdd_{name}.csv", index=True)


# write the little readme
with open(save_path+f"README_{name}.txt", 'w') as file:
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