import hydra
import torch
import pandas as pd
import os
import glob

from contrastive.utils.config import process_config
from contrastive.models.contrastive_learner_visualization import ContrastiveLearner_Visualization
from contrastive.data.datamodule import DataModule_Evaluation


def embeddings_to_pandas(embeddings, csv_path=None, verbose=False):
    # embeddings is the output of the compute_representations function
    # csv_path is the path where to save the csv. Set to None if you want to return the df
    columns_names = ['dim'+str(i+1) for i in range(embeddings[0].shape[1])]
    values = pd.DataFrame(embeddings[0].numpy(), columns=columns_names)
    labels = embeddings[1]
    labels = pd.DataFrame(labels, columns=['ID'])
    df_embeddings = pd.concat([labels, values], axis=1)
    df_embeddings = df_embeddings.groupby('ID').mean()  # remove one copy each ID
    if verbose:
        print("embeddings:", df_embeddings.iloc[:10,:])
        print("nb of elements:", df_embeddings.shape[0])
    if csv_path:
        df_embeddings.to_csv(csv_path)
    else:
        return df_embeddings


@hydra.main(config_name='config', config_path="configs")
def compute_embeddings(config):
    config = process_config(config)

    config.mode = 'evaluation'

    # create new models in mode visualisation
    data_module = DataModule_Evaluation(config)
    data_module.setup(stage='validate')

    model = ContrastiveLearner_Visualization(config,
                               sample_data=data_module)

    # fetch and load weights
    paths = config.model_path+"/logs/*/version_0/checkpoints"+r'/*.ckpt'
    files = glob.glob(paths)
    print("model_weights:", files[0])
    cpkt_path = files[0]
    checkpoint = torch.load(cpkt_path, map_location=torch.device(config.device))

    model.load_state_dict(checkpoint['state_dict'])

    # create folder where to save the embeddings
    embeddings_path = config.embeddings_save_path
    os.makedirs(embeddings_path)

    # calculate embeddings for training set and save them somewhere
    print("TRAIN SET")
    train_embeddings = model.compute_representations(data_module.train_dataloader())

    # convert the embeddings to pandas df and save them
    embeddings_to_pandas(train_embeddings,
                         csv_path=embeddings_path+"train_embeddings.csv")

    # same thing for validation set
    print("VAL SET")
    val_embeddings = model.compute_representations(data_module.val_dataloader())

    embeddings_to_pandas(val_embeddings,
                         csv_path=embeddings_path+"val_embeddings.csv")

    # /!\ DOESN'T WORK ON TEST => problem in create_datasets
    # same thing for test set
    print("TEST SET")
    test_embeddings = model.compute_representations(data_module.test_dataloader())

    embeddings_to_pandas(test_embeddings,
                         csv_path=embeddings_path+"test_embeddings.csv")



if __name__ == "__main__":
    compute_embeddings()