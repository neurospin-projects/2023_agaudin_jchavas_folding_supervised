import hydra
import pandas as pd

@hydra.main(config_name='config', config_path="configs")
def train_classifier(config):
    
    # import the embeddings (supposed to be already computed)
    dir_path = config.embeddings_path
    train_embeddings = pd.read_csv(dir_path+'/train_embeddings.csv', index_col=0)
    val_embeddings = pd.read_csv(dir_path+'/val_embeddings.csv', index_col=0)
    test_embeddings = pd.read_csv(dir_path+'/test_embeddings.csv', index_col=0)
    # regroup them in one dataframe (discuss with Joël)
    embeddings = pd.concat([train_embeddings, val_embeddings, test_embeddings],
    axis=0, ignore_index=True)

    # get the labels (0 = no paracingulate, 1 = paracingulate)
    labels = pd.read_csv(config.labels_paracingulate)
    # supposed to contain ID as index and one column nammed label

    # join the two df (join allows to be sure to have the labels in the right order)
    full_df = embeddings.join(labels, on='ID')



