import pandas as pd
import os
from os.path import join
from cleaning import *


def get_amazon_data(data_path):
    """
    Read amazon data from train,
    dev and test directories

    Parameters
    ----------
        data_path: str
            The path of the folder
            which hold the data
    Returns
    -------
        train_data: Dataframe
            The training data
        val_data: Dataframe
            The validation data
        test_data: Dataframe
            The test data
    """
    train_data = pd.DataFrame(columns=['review_id', 'product_id',
                                       'reviewer_id', 'stars',
                                       'review_body', 'review_title',
                                       'language', 'product_category'])
    val_data = pd.DataFrame(columns=['review_id', 'product_id',
                                     'reviewer_id', 'stars',
                                     'review_body', 'review_title',
                                     'language', 'product_category'])
    test_data = pd.DataFrame(columns=['review_id', 'product_id',
                                      'reviewer_id', 'stars',
                                      'review_body', 'review_title',
                                      'language', 'product_category'])
    for dir in os.listdir(data_path):

        # get train data
        if dir == 'train':
            for elm in os.listdir(join(data_path, dir)):
                if elm.endswith('.csv'):
                    data = pd.read_csv(join(join(data_path, dir), elm))
                train_data = pd.concat([train_data, data], ignore_index=True)

        # get validation data
        if dir == 'dev':
            for elm in os.listdir(join(data_path, dir)):
                if elm.endswith('.csv'):
                    data = pd.read_csv(join(join(data_path, dir), elm))
                val_data = pd.concat([val_data, data], ignore_index=True)

        # get test data
        if dir == 'test':
            for elm in os.listdir(join(data_path, dir)):
                if elm.endswith('.csv'):
                    data = pd.read_csv(join(join(data_path, dir), elm))
                    test_data = pd.concat([test_data, data],
                                          ignore_index=True)

    train_data.drop(['review_id', 'product_id', 'reviewer_id',
                    'review_title', 'product_category'],
                    axis=1, inplace=True)
    train_data.stars.replace({1: 0, 2: 0, 4: 1, 5: 1},
                             inplace=True)
    train_data.drop(train_data.loc[train_data['stars'] == 3].index,
                    inplace=True)
    train_data.rename(columns={'stars': 'labels',
                               'review_body': 'reviews',
                               'language': 'lang'},
                      inplace=True)

    val_data.drop(['review_id', 'product_id', 'reviewer_id',
                  'review_title', 'product_category'],
                  axis=1, inplace=True)
    val_data.stars.replace({1: 0, 2: 0, 4: 1, 5: 1},
                           inplace=True)
    val_data.drop(val_data.loc[val_data['stars'] == 3].index,
                  inplace=True)
    val_data.rename(columns={'stars': 'labels',
                             'review_body': 'reviews',
                             'language': 'lang'},
                    inplace=True)

    test_data.drop(['review_id', 'product_id', 'reviewer_id',
                   'review_title', 'product_category'],
                   axis=1, inplace=True)
    test_data.stars.replace({1: 0, 2: 0, 4: 1, 5: 1},
                            inplace=True)
    test_data.drop(test_data.loc[test_data['stars'] == 3].index,
                   inplace=True)
    test_data.rename(columns={'stars': 'labels',
                              'review_body': 'reviews',
                              'language': 'lang'},
                     inplace=True)

    return train_data, val_data, test_data


if __name__ == '__main__':
    print(os.getcwd())
    data_path = join(os.getcwd(), 'data_transform/Sampled MARC')
    train_data, val_data, test_data = get_amazon_data(data_path)
    # Add the list of the languages.
    # allowed values ['fr', 'de', 'es', en]
    languages = ['fr', 'de', 'es']
    # get the train, validation and test data
    print('read data ...')
    train_data = train_data[train_data['lang'].isin(languages)]
    train_data['split'] = 'train'
    val_data = val_data[val_data['lang'].isin(languages)]
    val_data['split'] = 'val'
    test_data = test_data[test_data['lang'].isin(languages)]
    test_data['split'] = 'test'
    data = pd.concat([train_data, val_data, test_data], ignore_index=True)
    # Data cleaning and preprocessing
    data.reviews = data.reviews.map(lambda x: str(x).lower())
    remove_spec_char(data)
    data.split.replace({'val': 'train'}, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data = cleaning_fn(data)
    # Save the constructed data
    with open(r'../data/corpus/amazon.txt', 'w', encoding='utf-8') as file:
        for item in data.reviews:
            # write each item on a new line
            file.write("%s\n" % item)
    df = data[['split', 'labels']]
    df.to_csv('../data/amazon.txt', sep='\t', header=False)
