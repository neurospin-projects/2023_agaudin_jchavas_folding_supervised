import hydra
import torch
import pandas as pd
import glob

from contrastive.utils.config import process_config
from contrastive.models.contrastive_learner_visualization import ContrastiveLearner_Visualization
from contrastive.data.datamodule import DataModule_Evaluation


def embeddings_to_pandas(embeddings, csv_path=None):
    # embeddings is the output of the compute_representations function
    # csv_path is the path where to save the csv. Set to None if you want to return the df
    print(len(embeddings[1]))
    columns_names = ['dim'+str(i+1) for i in range(embeddings[0].shape[1])]
    values = pd.DataFrame(embeddings[0].numpy(), columns=columns_names)
    labels = embeddings[1]
    labels = pd.DataFrame(labels, columns=['ID'])
    df_embeddings = pd.concat([labels, values], axis=1)
    if csv_path:
        df_embeddings.to_csv(csv_path)
    else:
        return df_embeddings


@hydra.main(config_name='config', config_path="configs")
def compute_embeddings(config):
    config = process_config(config)

    data_module = DataModule_Evaluation(config)
    data_module.setup(stage='validate')

    model = ContrastiveLearner_Visualization(config,
                               sample_data=data_module)

    # /!\ model weights are not necessarly at the same place that the embeddings you want to use
    if config.model_path:
        paths = config.model_path+r'*.ckpt'
    else:
        paths = config.embeddings_path+"logs/lightning_logs/version_0/checkpoints/"+r'*.ckpt'
    files = glob.glob(paths)
    print("model_weights:", files[0])
    cpkt_path = files[0]
    checkpoint = torch.load(cpkt_path, map_location=torch.device(config.device))

    model.load_state_dict(checkpoint['state_dict'])

    # calculate embeddings for training set and save them somewhere
    train_embeddings = model.compute_representations(data_module.train_dataloader())
    print("train embeddings:", train_embeddings[0][:10])
    print(train_embeddings[0].shape)
    print(train_embeddings[1][:10])

    # convert the embeddings to pandas df and save them
    embeddings_to_pandas(train_embeddings,
                         csv_path=config.embeddings_path+"train_embeddings.csv")

    # same thing for validation set
    val_embeddings = model.compute_representations(data_module.val_dataloader())
    print("validation embeddings:",val_embeddings[0][:10])

    embeddings_to_pandas(val_embeddings,
                         csv_path=config.embeddings_path+"val_embeddings.csv")

    # /!\ DOESN'T WORK ON TEST
    # same thing for test set
    test_embeddings = model.compute_representations(data_module.test_dataloader())
    print("test embeddings:", test_embeddings[:10])

    embeddings_to_pandas(test_embeddings,
                         csv_path=config.embeddings_path+"test_embeddings.csv")



if __name__ == "__main__":
    compute_embeddings()