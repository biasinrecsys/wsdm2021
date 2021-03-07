#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def real(interactions, column, target=0.0):
    minority_interactions = interactions[interactions[column] > 0][['user_id', 'item_id']]
    majority_interactions = interactions[interactions[column] == 0][['user_id', 'item_id']]
    total_interaction = len(minority_interactions.index) + len(majority_interactions.index)
    print('original', total_interaction, 'minority', len(minority_interactions.index) / total_interaction, 'and majority', len(majority_interactions.index) / total_interaction)
    no_required_minority_interactions = (len(majority_interactions) * target) / (1-target) - len(minority_interactions.index)
    old_minority_interactions = minority_interactions.sample(frac=no_required_minority_interactions / len(minority_interactions.index), replace=True, random_state=1)
    minority_interactions = pd.concat([old_minority_interactions, minority_interactions])
    total_interaction = len(minority_interactions.index) + len(majority_interactions.index)
    print('upsampled', total_interaction, 'minority', len(minority_interactions.index) / total_interaction, 'and majority', len(majority_interactions.index) / total_interaction)
    return pd.concat([majority_interactions, minority_interactions], sort=True)

def fake(interactions, column, items, target=0.0):
    minority_interactions = interactions[interactions[column] > 0][['user_id', 'item_id']]
    majority_interactions = interactions[interactions[column] == 0][['user_id', 'item_id']]
    total_interaction = len(minority_interactions.index) + len(majority_interactions.index)
    print('original', total_interaction, 'minority', len(minority_interactions.index) / total_interaction, 'and majority', len(majority_interactions.index) / total_interaction)
    no_required_minority_interactions = int((len(majority_interactions) * target) / (1-target) - len(minority_interactions.index))
    umapping = {index:list(set(items)-set(group['item_id'].values)) for index, group in interactions.groupby(by='user_id')}
    new_fake_interactions = []
    for i, u in enumerate(np.random.choice(np.unique(interactions['user_id'].values), no_required_minority_interactions)):
        if (i+1) % 100000 == 0:
            print('\rcomputing', i+1, 'of', no_required_minority_interactions, end='')
        new_item = np.random.choice(umapping[int(u)])
        new_fake_interactions.append([int(u), new_item])
        umapping[int(u)].remove(new_item)
    minority_interactions = pd.concat([minority_interactions, pd.DataFrame(new_fake_interactions, columns=['user_id', 'item_id'])])
    total_interaction = len(minority_interactions.index) + len(majority_interactions.index)
    print('\nupsampled', total_interaction, 'minority', len(minority_interactions.index) / total_interaction, 'and majority', len(majority_interactions.index) / total_interaction)
    return pd.concat([majority_interactions, minority_interactions], sort=True)

def fakeByPop(interactions, column, items, target=0.0):
    minority_interactions = interactions[interactions[column] > 0][['user_id', 'item_id']]
    majority_interactions = interactions[interactions[column] == 0][['user_id', 'item_id']]
    item_popularity = minority_interactions.groupby(by='item_id').count()['user_id']
    total_interaction = len(minority_interactions.index) + len(majority_interactions.index)
    print('original', total_interaction, 'minority', len(minority_interactions.index) / total_interaction, 'and majority', len(majority_interactions.index) / total_interaction)
    no_required_minority_interactions = int((len(majority_interactions) * target) / (1-target) - len(minority_interactions.index))
    umapping = {index:list(set(items)-set(group['item_id'].values)) for index, group in interactions.groupby(by='user_id')}
    item_probability = item_popularity / np.sum(item_popularity)
    pmapping = {u:(item_probability[umapping[int(u)]] / np.sum(item_probability[umapping[int(u)]])) for u, _ in interactions.groupby(by='user_id')}
    new_fake_interactions = []
    for i, u in enumerate(np.random.choice(np.unique(interactions['user_id'].values), no_required_minority_interactions)):
        print('computing', i, 'of', no_required_minority_interactions)
        new_item = np.random.choice(umapping[int(u)], p=pmapping[u])
        new_fake_interactions.append([int(u), new_item])
        umapping[int(u)].remove(new_item)
        pmapping[u] = item_probability[umapping[int(u)]] / np.sum(item_probability[umapping[int(u)]])
    minority_interactions = pd.concat([minority_interactions, pd.DataFrame(new_fake_interactions, columns=['user_id', 'item_id'])])
    total_interaction = len(minority_interactions.index) + len(majority_interactions.index)
    print('\nupsampled', total_interaction, 'minority', len(minority_interactions.index) / total_interaction, 'and majority', len(majority_interactions.index) / total_interaction)
    return pd.concat([majority_interactions, minority_interactions], sort=True)
