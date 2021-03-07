#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math
import sys

from helpers.utils import save_obj, get_bpr_loss

class Model:

    def __init__(self, users, items, observed_relevance, unobserved_relevance, category_per_item, item_field, user_field, rating_field):

        print('Initializing user, item, and categories lists')
        self.name = 'Unknown'
        self.users = users
        self.items = items
        self.categories = list(np.unique(category_per_item))
        self.no_users = len(users)
        self.no_items = len(items)
        self.no_categories = len(self.categories)
        self.no_interactions = len(observed_relevance.index)

        print('Initializing observed, unobserved, and predicted relevance scores')
        self.observed_relevance = self.__get_feedback(observed_relevance, item_field, user_field, rating_field)
        self.unobserved_relevance = self.__get_feedback(unobserved_relevance, item_field, user_field, rating_field)
        self.predicted_relevance = None

        print('Initializing item popularity lists')
        self.item_popularity = np.array([len(observed_relevance[observed_relevance[item_field]==item_id].index) for item_id in items])

        print('Initializing category per item')
        self.category_per_item = category_per_item

        print('Initializing category preference per user')
        self.categories_per_user = self.__get_representation()

        print('Initializing metrics')
        self.metrics = None

    def __get_feedback(self, feedback, item_field, user_field, rating_field):
        relevance = np.zeros((self.no_users, self.no_items))
        relevance[np.array(feedback[user_field].tolist()), np.array(feedback[item_field].tolist())] = np.array(feedback[rating_field].tolist())
        return relevance

    def __get_representation(self):
        categories_per_user = np.zeros((self.no_users, self.no_categories))

        for user_id, user_observed_relevance in enumerate(self.observed_relevance):
            user_ubserved_items =  np.nonzero(user_observed_relevance)[0]
            user_observed_categories = [self.category_per_item[item_id] for item_id in user_ubserved_items]
            for category in user_observed_categories:
                categories_per_user[user_id, self.categories.index(category)] += 1
            categories_per_user[user_id] = categories_per_user[user_id] / np.sum(categories_per_user[user_id])

        return categories_per_user

    def __get_dcg(self, y_true, y_score, k, gains='exponential'):
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:k])
        if gains == 'exponential':
            gains = 2 ** y_true - 1
        elif gains == 'linear':
            gains = y_true
        else:
            raise ValueError('Invalid gains option')
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(np.array(gains) / np.array(discounts))

    def predict(self):
        self.predicted_relevance = None

    def train(self):
        self.model = None
        self.history = None

    def test(self, item_group=None, cutoffs=np.array([10]), gains='exponential'):
        self.metrics = {}
        self.metrics['precision'] = np.zeros((len(cutoffs),self.no_users))
        self.metrics['recall'] = np.zeros((len(cutoffs), self.no_users))
        self.metrics['ndcg'] = np.zeros((len(cutoffs), self.no_users))
        self.metrics['hit'] = np.zeros((len(cutoffs), self.no_users))
        self.metrics['mean_popularity'] = np.zeros((len(cutoffs), self.no_users))
        self.metrics['diversity'] = np.zeros((len(cutoffs), self.no_users))
        self.metrics['novelty'] = np.zeros((len(cutoffs), self.no_users))
        self.metrics['item_coverage'] = np.zeros((len(cutoffs), self.no_items))

        if item_group is not None:
            self.metrics['visibility'] = np.zeros((len(cutoffs), self.no_users))
            self.metrics['exposure'] = np.zeros((len(cutoffs), self.no_users))

        for user_id, (user_observed, user_unobserved, user_predicted) in enumerate(zip(self.observed_relevance, self.unobserved_relevance, self.predicted_relevance)):

            if (user_id % 1000) == 0:
                print('\rComputing metrics for user', user_id, '/', self.no_users, end='')

            if user_id < self.no_users:

                train_pids = np.nonzero(user_observed)[0]
                test_pids = np.nonzero(user_unobserved)[0]
                y_true = np.zeros(self.no_items, dtype=np.int32)
                y_true[test_pids] = 1.0
                y_pred = user_predicted
                y_pred[train_pids] = -math.inf

                for index_k, k in enumerate(cutoffs):
                    top_k = np.argsort(-y_pred)[:k]
                    self.metrics['precision'][index_k,user_id] = len(set(top_k) & set(test_pids)) / float(k)
                    self.metrics['recall'][index_k,user_id] = len(set(top_k) & set(test_pids)) / (len(set(test_pids)) + sys.float_info.epsilon)
                    self.metrics['ndcg'][index_k,user_id] = self.__get_dcg(y_true, y_pred, k, gains) / (self.__get_dcg(y_true, y_true, k, gains) + sys.float_info.epsilon)
                    self.metrics['hit'][index_k, user_id] = (1 if len(set(top_k) & set(test_pids)) > 0 else 0)
                    self.metrics['mean_popularity'][index_k,user_id] = np.mean(self.item_popularity[top_k])
                    for pos, item_id in enumerate(top_k):
                        if item_group is not None:
                            self.metrics['exposure'][index_k, user_id] += (1/math.log(pos+1+1) if item_group[item_id] == 0 else 0)
                            self.metrics['visibility'][index_k,user_id] += (1-item_group[item_id])
                        self.metrics['item_coverage'][index_k,item_id] += 1
                        self.metrics['novelty'][index_k,user_id] += -math.log(self.item_popularity[item_id] / (self.no_users) + sys.float_info.epsilon, 2)
                    self.metrics['novelty'][index_k,user_id] = self.metrics['novelty'][index_k,user_id] / k
                    self.metrics['diversity'][index_k,user_id] = len(np.unique([self.category_per_item[item_id] for item_id in top_k])) / self.no_categories
                    if item_group is not None:
                        self.metrics['exposure'][index_k, user_id] /= np.sum([1/math.log(pos+1+1) for pos in range(k)])
                        self.metrics['visibility'][index_k, user_id] /= k
        print('\rComputing metrics for user', user_id+1, '/', self.no_users)

    def show_metrics(self, index_k=0):
        precision = round(np.mean([v for v in self.metrics['precision'][index_k, :self.no_users]]), 4)
        recall = round(np.mean([v for v in self.metrics['recall'][index_k, :self.no_users]]), 4)
        ndcg = round(np.mean([v for v in self.metrics['ndcg'][index_k, :self.no_users]]), 4)
        hit = round(np.sum([v for v in self.metrics['hit'][index_k, :self.no_users]]) / self.no_users, 4)
        avgpop = round(np.mean([v for v in self.metrics['mean_popularity'][index_k, :self.no_users]]), 4)
        diversity = round(np.mean([v for v in self.metrics['diversity'][index_k, :self.no_users]]), 4)
        novelty = round(np.mean([v for v in self.metrics['novelty'][index_k, :self.no_users]]), 4)
        item_coverage = round(len([1 for m in self.metrics['item_coverage'][index_k] if m > 0]) / self.no_items, 2)
        user_coverage = round(len([1 for v in self.metrics['precision'][index_k, :self.no_users] if v != 0]) / self.no_users, 4)
        print('Precision:', precision, '\nRecall:', recall, '\nNDCG:', ndcg, '\nHit Rate:', hit, '\nAvg Popularity:', avgpop, '\nCategory Diversity:', diversity, '\nNovelty:', novelty, '\nItem Coverage:', item_coverage, '\nUser Coverage:', user_coverage)
        if 'exposure' in self.metrics:
            exp = round(np.mean(self.metrics['exposure'][index_k, :self.no_users]), 4)
            print('Minority Exposure:', exp)
        if 'visibility' in self.metrics:
            vis = round(np.mean(self.metrics['visibility'][index_k, :self.no_users]), 4)
            print('Minority Visibility:', vis)

    def print(self):
        if self.model:
            self.model.summary()

    def get_predictions(self):
        return self.predicted_relevance

    def get_metrics(self):
        return self.metrics

    def get_model(self, model_path):
        if self.model is not None:
            self.model.save(model_path)
            save_obj(self.history, model_path.split('.')[0] + '.hist')

    def set_predictions(self, predicted_relevance):
        self.predicted_relevance = predicted_relevance

    def set_metrics(self, metrics):
        self.metrics = metrics

    def set_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path, custom_objects={'get_bpr_loss': get_bpr_loss})