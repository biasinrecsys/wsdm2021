#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random

def generator(observed_relevance, categories, no_categories, category_per_item, categories_per_user, no_negatives=10, gen_mode='point', item_popularity=None):
    user_input, user_attr, item_i_input, item_i_attr, item_j_input, item_j_attr, labels = [], [], [], [], [], [], []
    no_users, no_items = observed_relevance.shape[0], observed_relevance.shape[1]

    users, items = np.nonzero(observed_relevance)
    positive_set_list = [set() for _ in range(no_users)]
    for (user_id, item_id) in zip(users, items):
        positive_set_list[int(user_id)].add(int(item_id))

    negative_set_list = [set() for _ in range(no_users)]
    for user_id in range(no_users):
        negative_set_list[user_id] = list(set(range(no_items)) - set(positive_set_list[int(user_id)]))

    for index, (user_id, item_id) in enumerate(zip(users, items)):
        if (index % 100000) == 0:
            print('\rComputing instances for interaction', index, '/', len(users), 'of type', gen_mode, end='')

        if gen_mode == 'point':
            user_input.append(user_id)
            item_i_input.append(item_id)
            labels.append(1)

            for _ in range(no_negatives):
                user_input.append(user_id)
                item_i_input.append(random.choice(negative_set_list[user_id]))
                labels.append(0)

        elif gen_mode == 'pair':
            for _ in range(no_negatives):
                user_input.append(user_id)
                item_i_input.append(item_id)
                item_j_input.append(random.choice(negative_set_list[user_id]))
                labels.append(item_popularity[item_id] / np.sum(item_popularity))

        else:
            raise NotImplementedError('The generation type ' + gen_mode + ' is not implemented.')
    print()

    return (np.array(user_input), np.array(item_i_input),np.array(item_j_input)), (np.array(labels))


def balanced_generator(observed_relevance, categories, no_categories, category_per_item, categories_per_user, no_negatives=10, popularity_win=1500, item_popularity=None):
    user_input, user_attr, item_i_input, item_i_attr, item_j_input, item_j_attr, labels, pops = [], [], [], [], [], [], [], []
    no_users, no_items = observed_relevance.shape[0], observed_relevance.shape[1]

    users, items = np.nonzero(observed_relevance)
    positive_set_list = [set() for _ in range(no_users)]
    for (user_id, item_id) in zip(users, items):
        positive_set_list[int(user_id)].add(int(item_id))

    negative_set_list = [set() for _ in range(no_users)]
    for user_id in range(no_users):
        negative_set_list[user_id] = list(set(range(no_items)) - set(positive_set_list[int(user_id)]))

    item_id_rank = np.array(np.argsort(item_popularity)[::-1])

    mapping_max = {v: set(item_id_rank[max(0,i-popularity_win):i]) for i, v in enumerate(item_id_rank)}
    mapping_min = {v: set(item_id_rank[i:min(no_items, i+popularity_win)]) for i, v in enumerate(item_id_rank)}

    most_pop_flag = False
    for index, (user_id, item_id) in enumerate(zip(users, items)):

        if (index % 100000) == 0:
            print('\rComputing instances for interaction', index, '/', len(users), 'of type pair balanced', end='')

        for _ in range(no_negatives):
            user_input.append(user_id)
            item_i_input.append(item_id)

            if most_pop_flag:
                more_popular_items = list(mapping_max[item_id] - positive_set_list[user_id])
                another_item = random.choice(more_popular_items if len(more_popular_items) > 0 else (negative_set_list[int(user_id)]))
            else:
                less_popular_items = list(mapping_min[item_id] - positive_set_list[user_id])
                another_item = random.choice(less_popular_items if len(less_popular_items) > 0 else (negative_set_list[int(user_id)]))

            most_pop_flag = (not most_pop_flag)

            item_j_input.append(another_item)

            labels.append(item_popularity[item_id] / np.sum(item_popularity))

    print()

    return (np.array(user_input), np.array(item_i_input), np.array(item_j_input)), (np.array(labels))