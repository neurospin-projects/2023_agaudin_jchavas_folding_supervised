import pandas as pd
from datetime import datetime

from contrastive.utils.models_database import *


dataset = 'cingulate_ACCpatterns'

## construct the database
# folders to look for the models in
folders = ["/neurospin/dico/agaudin/Runs/04_pointnet/Output", "/neurospin/dico/agaudin/Runs/03_monkeys/Output/analysis_folders/convnet",
"/neurospin/dico/agaudin/Runs/03_monkeys/Output/analysis_folders/densenet2", "/neurospin/dico/agaudin/Runs/03_monkeys/Output/convnet_exploration"]
bdd = []
visited = []

generate_bdd_models(folders, bdd, visited, verbose=False, dataset=dataset)

bdd = pd.DataFrame(bdd)
print("Number of subjects:", bdd.shape[0])

# remove useless columns
bdd = post_process_bdd_models(bdd, hard_remove=["partition", "patch_size", "block_config", "numpy_all"], git_branch=True)


# save the database
save_path = "/neurospin/dico/agaudin/Runs/new_bdd_models.csv"
bdd.to_csv(save_path, index=True)


# write the little readme
with open("/neurospin/dico/agaudin/Runs/new_readme_bdd.txt", 'w') as file:
    file.write("Contient les paramètres de tous les modèles d'intérêt (dossiers précisés en-dessous). La base est faite en sorte que \
seuls les paramètres qui changent entre les modèles soient enregistrés.\n")
    file.write("\n")
    file.write("Généré avec contrastive/notebooks/generate_bdd.ipynb le " + datetime.now().strftime('%d/%m/%Y à %H:%M') + '.\n')
    file.write("\n")
    file.write("Dossiers utilisés : [")
    for folder in folders:
        file.write(folder)
        if folder == folders[-1]:
            file.write(']')
        else:
            file.write(',\n')