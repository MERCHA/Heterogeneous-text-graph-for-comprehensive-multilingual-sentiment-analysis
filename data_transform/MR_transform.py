import re

import pandas as pd
from datasets import (get_dataset_split_names, load_dataset,
                      load_dataset_builder)

from cleaning import *

# datasets_names = ["muchocine", "allocine", "imdb"]

# ds_builder = load_dataset_builder("muchocine")
# print(ds_builder.info.description)
# print(ds_builder.info.features)
# print(get_dataset_split_names("muchocine"))


def read_muchocine():
    """
    Load the muchocine dataset using
    the datasets from Hugging face
    """

    muchocine_dataset = load_dataset("muchocine", split="train")
    muchocine_df = pd.DataFrame(muchocine_dataset)
    muchocine_df.drop(columns=['review_summary'], inplace=True)
    muchocine_df.star_rating.replace({1: 0, 2: 0, 4: 1, 5: 1},
                             inplace=True)
    muchocine_df.drop(muchocine_df.loc[muchocine_df['star_rating'] == 3].index,
                      inplace=True)
    muchocine_df.rename(columns={'review_body': 'review',
                                 'star_rating': 'label'},
                        inplace=True)
    return muchocine_df


def read_allocine():
    """
    Load the allocine dataset using
    the datasets from Hugging face
    """
    allocine_tr = load_dataset("allocine", split="train")
    allocine_tr = pd.DataFrame(allocine_tr)
    allocine_val = load_dataset("allocine", split="validation")
    allocine_val = pd.DataFrame(allocine_val)
    allocine_test = load_dataset("allocine", split="test")
    allocine_test = pd.DataFrame(allocine_test)

    return allocine_tr, allocine_val, allocine_test


def read_imdb():
    """
    Load the IMDB dataset using
    the datasets from Hugging face
    """
    imdb_tr = load_dataset("imdb", split="train")
    imdb_tr = pd.DataFrame(imdb_tr)
    imdb_tr.label.replace({'neg': 0, 'pos': 1},
                          inplace=True)
    imdb_tr.rename(columns={'text': 'review'},
                   inplace=True)
    imdb_test = load_dataset("imdb", split="test")
    imdb_test = pd.DataFrame(imdb_test)
    imdb_test.label.replace({'neg': 0, 'pos': 1},
                            inplace=True)
    imdb_test.rename(columns={'text': 'review'},
                     inplace=True)

    return imdb_tr, imdb_test

if __name__ == "__main__":
    muchocine_ds = read_muchocine()
    muchocine_ds = muchocine_ds.sample(frac=1).reset_index(drop=True)

    muchocine_pos = muchocine_ds[muchocine_ds.label == 1]
    muchocine_neg = muchocine_ds[muchocine_ds.label == 0]
    # data balancing
    muchocine_neg = muchocine_neg.sample(n=len(muchocine_pos))
    # spliting pos data
    tr_muchocine_pos = muchocine_pos[0:323]
    val_muchocine_pos = muchocine_pos[323:392]
    test_muchocine_pos = muchocine_pos[392:461]
    # spliting neg data
    tr_muchocine_neg = muchocine_pos[0:323]
    val_muchocine_neg = muchocine_pos[323:392]
    test_muchocine_neg = muchocine_pos[392:461]

    train_muchocine = pd.concat([tr_muchocine_pos, tr_muchocine_neg], ignore_index=True)
    train_muchocine['split'] = 'train'
    val_muchocine = pd.concat([val_muchocine_pos, val_muchocine_neg], ignore_index=True)
    val_muchocine['split'] = 'val'
    test_muchocine = pd.concat([test_muchocine_pos, test_muchocine_neg], ignore_index=True)
    test_muchocine['split'] = 'test'

    # \\\ data balancing
    allocine_tr, allocine_val, allocine_test = read_allocine()
    train_allocine = pd.concat([allocine_tr[allocine_tr.label == 0].sample(n=2500),
                                allocine_tr[allocine_tr.label == 1].sample(n=2500)],
                               ignore_index=True)
    train_allocine['split'] = 'train'
    val_allocine = pd.concat([allocine_val[allocine_val.label == 0].sample(n=500),
                              allocine_val[allocine_val.label == 1].sample(n=500)],
                             ignore_index=True)
    val_allocine['split'] = 'val'
    test_allocine = pd.concat([allocine_test[allocine_test.label == 0].sample(n=500),
                               allocine_test[allocine_test.label == 1].sample(n=500)],
                              ignore_index=True)
    test_allocine['split'] = 'test'

    imdb_tr, imdb_test = read_imdb()
    train_imdb = pd.concat([imdb_tr[imdb_tr.label == 0].sample(n=2500),
                            imdb_tr[imdb_tr.label == 1].sample(n=2500)], ignore_index=True)
    train_imdb['split'] = 'train'
    val_imdb = pd.concat([imdb_test[imdb_test.label == 0][0:500],
                          imdb_test[imdb_test.label == 1][0:500]], ignore_index=True)
    val_imdb['split'] = 'val'
    test_imdb = pd.concat([imdb_test[imdb_test.label == 0][500:1000],
                           imdb_test[imdb_test.label == 1][500:1000]], ignore_index=True)
    test_imdb['split'] = 'test'

    train_data = pd.concat([train_muchocine, train_allocine, train_imdb], ignore_index=True)
    val_data = pd.concat([val_muchocine, val_allocine, val_imdb], ignore_index=True)
    test_data = pd.concat([test_muchocine, test_allocine, test_imdb], ignore_index=True)

    # print(train_data.info())
    # print(val_data.info())
    # print(test_data.info())

    MR_dataset = pd.concat([train_data, val_data, test_data], ignore_index=True)
    MR_dataset.reset_index(drop=True, inplace=True)


    # Data cleaning and preprocessing
    MR_dataset.review = MR_dataset.review.map(lambda x: str(x).lower())
    MR_dataset.review = MR_dataset.review.map(lambda x: remove_repeated_characters(str(x)))
    MR_dataset.review = MR_dataset.review.map(lambda x: remove_url(str(x)))
    MR_dataset.review = MR_dataset.review.map(lambda x: remove_emails(str(x)))
    MR_dataset.review = MR_dataset.review.map(lambda x: remove_htmltags(str(x)))
    # MR_dataset.review = MR_dataset.review.map(lambda x: remove_numbers(str(x)))
    remove_spec_char(MR_dataset)
    MR_dataset.review = MR_dataset.review.map(lambda x: remove_extra_space(str(x)))

    MR_dataset.to_csv('data_transform\Movie reviews\dataset.csv', index=False)

    MR_dataset.split.replace({'val': 'train'}, inplace=True)
    MR_dataset.reset_index(drop=True, inplace=True)

    with open(r'../data/corpus/MR.txt', 'w', encoding='utf-8') as file:
        for item in MR_dataset.review:
            # write each item on a new line
            file.write("%s\n" % item)

    df = MR_dataset[['split', 'label']]
    df.to_csv('../data/MR.txt', sep='\t', header=False)
