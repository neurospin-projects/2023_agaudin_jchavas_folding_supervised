import hydra
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from contrastive.utils.config import process_config


@hydra.main(config_name='config_no_save', config_path="../configs")
def compute_embeddings(config):
    config = process_config(config)

    if config.pca_Xtransform:
        pca_Xtransform = config.pca_Xtransform
    else:
        pca_Xtransform = config.pca_Xfit
        print("The pca will be applied to the same data it was trained on.")

    # train the pca
    X_fit = np.load(config.pca_Xfit)
    print("X_fit shape", X_fit.shape)

    # flatten X but the first dimension (ie the number of subjects)
    X_flatten = X_fit.reshape(X_fit.shape[0], np.prod(X_fit.shape[1:]))

    n_pca = config.n_pca

    pca = PCA(n_components=n_pca)
    pca.fit(X_flatten)

    # generate embeddings with the trained pca
    X_transform = np.load(pca_Xtransform)
    print("X_transform shape", X_transform.shape)
    X_transform = X_transform.reshape(
        X_transform.shape[0], np.prod(X_transform.shape[1:]))
    subjects_names = pd.read_csv(pca_Xtransform[:-4]+"_subject.csv")

    X_transform = pca.transform(X_transform)
    X_transform = pd.DataFrame(
        X_transform,
        columns=['dim'+str(i+1) for i in range(n_pca)])  # convert to df

    # add labels
    X_transform = pd.concat([subjects_names, X_transform], axis=1)

    # save embeddings
    save_path = config.embeddings_save_path+f"/pca_embeddings_{n_pca}.csv"
    print("save path:", save_path)
    X_transform.to_csv(save_path, index=False)

    with open(config.embeddings_save_path+f'/embeddings_info_{n_pca}.txt', 'w') \
            as file:
        file.write(f"fit embeddings: {config.pca_Xfit}\n")
        file.write(f"transform embeddings: {pca_Xtransform}\n")
        file.write(f"pca output dimension: {n_pca}")


if __name__ == "__main__":
    compute_embeddings()
