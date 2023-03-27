import pandas as pd
from datetime import datetime

from contrastive.utils.models_database import *


dataset = 'cingulate_ACCpatterns_1'

## construct the database
folders = ["/neurospin/dico/agaudin/Runs/05_rigourous/Output/nb_epochs"]
bdd = []
visited = []

# parameter to use or not the model with the best validation loss
best_model=False

generate_bdd_models(folders, bdd, visited, verbose=False, dataset=dataset, best_model=best_model)

bdd = pd.DataFrame(bdd)
print("Number of subjects:", bdd.shape[0])

for col in bdd.columns:
    print(col, bdd[col][0])

# remove useless columns
bdd = post_process_bdd_models(bdd, hard_remove=["partition", "patch_size"], git_branch=True)
# the hard remove are the ones containing [] char in their fields. They are (for now) patch_size, partition, numpy_all


# save the database
name = "nb_epochs"
save_path = "/neurospin/dico/agaudin/Runs/05_rigourous/Output/nb_epochs/"
bdd.to_csv(save_path+f"bdd_{name}.csv", index=True)


# write the little readme
with open(save_path+f"README_{name}.txt", 'w') as file:
    file.write("Contient les paramètres de tous les modèles d'intérêt (dossiers précisés en-dessous). La base est faite en sorte que \
seuls les paramètres qui changent entre les modèles soient enregistrés.\n")
    if best_model:
        file.write("\n")
        file.write("The given values are for the 'best models', ie the models saved when the validation loss is the lowest during training.\n")
    file.write("\n")
    file.write(f"Peformances données pour le dataset {dataset}\n")
    file.write("\n")
    file.write("Généré avec contrastive/evaluation/generate_bdd.py le " + datetime.now().strftime('%d/%m/%Y à %H:%M') + '.\n')
    file.write("\n")
    file.write("Dossiers utilisés : [")
    for folder in folders:
        file.write(folder)
        if folder == folders[-1]:
            file.write(']')
        else:
            file.write(',\n')