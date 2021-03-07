#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy import spatial
import tensorflow as tf
import numpy as np
import math
import sys

from helpers.utils import compute_weights
from models.model import Model

class RankerXQuad(Model):

    def __init__(self, users, items, observed_relevance, unobserved_relevance, category_per_item, item_field, user_field, rating_field):
        super().__init__(users, items, observed_relevance, unobserved_relevance, category_per_item, item_field, user_field, rating_field)

    def rerank(self, type='smooth', lmbda=0.4, k=10, rmax=100, head_tail_split=280):
        tail_head = np.argsort(self.item_popularity / np.sum(self.item_popularity))[::-1][:head_tail_split]
        tail_long = np.argsort(self.item_popularity / np.sum(self.item_popularity))[::-1][head_tail_split:]
        assert len(tail_head) + len(tail_long) == self.no_items

        for user_id, user_observed in zip(self.users, self.observed_relevance):
            print('\rPerforming reranking for user', user_id, '/', self.no_users, end='')
            user_scores = self.predicted_relevance[user_id]
            user_scores = (user_scores - min(user_scores)) / (max(user_scores) - min(user_scores))
            train_pids = np.nonzero(user_observed)[0]
            user_scores[train_pids] = -10000
            list_u = []
            p_d_u = [np.sum(user_observed[tail_head]) / np.sum(user_observed), np.sum(user_observed[tail_long]) / np.sum(user_observed)]
            assert np.sum(p_d_u) == 1
            self.predicted_relevance[user_id] = np.zeros(self.no_items)
            most_relevant_items = np.argsort(-user_scores)[:rmax]
            while len(list_u) < k:
                kwargs = {"tail_head": tail_head, "tail_long": tail_long, "mrtype": type, "list_u": list_u, "p_d_u": p_d_u}
                weights = np.apply_along_axis(compute_weights, 0, np.expand_dims(most_relevant_items, axis=0), **kwargs)
                comb_scores = np.array([(1 - lmbda) * user_scores[item_id] + lmbda * weights[item_pos] for item_pos, item_id in enumerate(most_relevant_items)])
                list_u.append(most_relevant_items[np.argsort(comb_scores)[-1]])
                self.predicted_relevance[user_id, list_u[-1]] = (k - len(list_u)) / k
                most_relevant_items = np.delete(most_relevant_items, np.argsort(comb_scores)[-1])
