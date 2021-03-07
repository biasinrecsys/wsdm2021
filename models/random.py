#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from models.model import Model

class Random(Model):

    def __init__(self, users, items, observed_relevance, unobserved_relevance, category_per_item, item_field, user_field, rating_field):
        super().__init__(users, items, observed_relevance, unobserved_relevance, category_per_item, item_field, user_field, rating_field)

    def predict(self):
        self.predicted_relevance = np.zeros((self.no_users, self.no_items))
        print('Computing predictions')
        for user_id in self.users:
            self.predicted_relevance[user_id] = np.random.uniform(0, 1, self.no_items)
