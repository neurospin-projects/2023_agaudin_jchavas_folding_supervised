import hydra
import torch
import pandas as pd

from contrastive.utils.config import process_config
from contrastive.models.contrastive_learner import ContrastiveLearner
from contrastive.models.contrastive_learner_visualization import ContrastiveLearner_Visualization
from contrastive.data.datamodule import DataModule_Evaluation


def embeddings_to_pandas(embeddings, csv_path):
    # embeddings is the output of the compute_representations function
    # csv_path is the path where to save the csv
    print(len(embeddings[1]))
    values = pd.DataFrame(embeddings[0].numpy(), columns=['dim1', 'dim2',\
'dim3', 'dim4'])
    labels = embeddings[1]
    labels = pd.DataFrame(labels, columns=['ID'])
    df_embeddings = pd.concat([labels, values], axis=1)
    df_embeddings.to_csv(csv_path)


@hydra.main(config_name='config', config_path="configs")
def compute_embeddings(config):
    config = process_config(config)

    data_module = DataModule_Evaluation(config)
    data_module.setup(stage='validate')

    model = ContrastiveLearner_Visualization(config,
                               sample_data=data_module)


    checkpoint = torch.load("/neurospin/dico/agaudin/Runs/02_explicabilite_humains_2022/Output/\
2022-05-18/11-00-10/logs/lightning_logs/version_0/checkpoints/epoch=299-step=8400.ckpt",
map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['state_dict'])

    # calculate embeddings for training set and save them somewhere
    train_embeddings = model.compute_representations(data_module.train_dataloader())
    print("train embeddings:", train_embeddings[0][:10])
    print(train_embeddings[0].shape)
    print(train_embeddings[1][:10])

    # convert the embeddings to pandas df and save them
    embeddings_to_pandas(train_embeddings, "/neurospin/dico/agaudin/Runs/02_explicabilite_humains_2022/\
Output/2022-05-18/11-00-10/train_embeddings.csv")

    # same thing for validation set
    val_embeddings = model.compute_representations(data_module.val_dataloader())
    print("validation embeddings:",val_embeddings[0][:10])

    embeddings_to_pandas(val_embeddings, "/neurospin/dico/agaudin/Runs/02_explicabilite_humains_2022/\
Output/2022-05-18/11-00-10/val_embeddings.csv")

    # /!\ DOESN'T WORK ON TEST
    # same thing for test set
    test_embeddings = model.compute_representations(data_module.test_dataloader())
    print("test embeddings:", test_embeddings[:10])

    embeddings_to_pandas(test_embeddings, "/neurospin/dico/agaudin/Runs/02_explicabilite_humains_2022/\
Output/2022-05-18/11-00-10/test_embeddings.csv")



if __name__ == "__main__":
    compute_embeddings()