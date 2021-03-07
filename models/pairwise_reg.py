#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os

from helpers.instances_creator import generator, balanced_generator
from helpers.utils import load_obj, save_obj
from models.model import Model
from helpers.utils import get_bpr_loss, get_dot_difference, get_dot_difference_shape, get_correlation_loss

class PairWise(Model):

    def __init__(self, users, items, observed_relevance, unobserved_relevance, category_per_item, item_field, user_field, rating_field):
        super().__init__(users, items, observed_relevance, unobserved_relevance, category_per_item, item_field, user_field, rating_field)

    def __get_model(self, mf_dim=10):
        user_embedding = tf.keras.layers.Embedding(self.no_users + 1, mf_dim, name='UserEmb')
        item_embedding = tf.keras.layers.Embedding(self.no_items + 1, mf_dim, name='ItemEmb')

        user_input = tf.keras.layers.Input(shape=[1], name='UserInput')
        user_vec = tf.keras.layers.Flatten(name='FlatUserEmb')(user_embedding(user_input))

        i_item_input = tf.keras.layers.Input(shape=[1], name='PosItemInput')
        pos_item_vec = tf.keras.layers.Flatten(name='FlatPosItemEmb')(item_embedding(i_item_input))

        j_item_input = tf.keras.layers.Input(shape=[1], name='NegItemInput')
        neg_item_vec = tf.keras.layers.Flatten(name='FlatNegItemEmb')(item_embedding(j_item_input))

        dot_difference = tf.keras.layers.Lambda(get_dot_difference, output_shape=get_dot_difference_shape, name='acc')([user_vec, pos_item_vec, neg_item_vec])
        dot_other_dot_difference = tf.keras.layers.Lambda(get_dot_difference, output_shape=get_dot_difference_shape, name='corr')([user_vec, pos_item_vec, neg_item_vec])

        return tf.keras.Model(inputs=[user_input, i_item_input, j_item_input], outputs=[dot_difference, dot_other_dot_difference])

    def train(self, rweight=0.0, no_epochs=100, batches=1024, lr=0.001, no_factors=10, no_negatives=10, gen_mode='pair', val_split=0.01, val_interval=4):

        print('Generating training instances', 'of type', gen_mode)

        print('Created training instances randomly')
        x, y = generator(self.observed_relevance, self.categories, self.no_categories, self.category_per_item, self.categories_per_user, no_negatives=no_negatives, gen_mode=gen_mode, item_popularity=self.item_popularity)

        print('Performing training -', 'Epochs', no_epochs, 'Batch Size', batches, 'Learning Rate', lr, 'Factors', no_factors, 'Negatives', no_negatives, 'Mode', gen_mode)
        self.model = self.__get_model()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss=[get_bpr_loss, get_correlation_loss], loss_weights=[1-rweight, rweight])

        user_input, item_i_input, item_j_input = x
        labels = y

        train_instance_indexes = np.random.choice(list(range(len(user_input))), int(len(user_input) * (1-val_split)), replace=False)
        val_instance_indexes = np.array(list(set(range(len(user_input))) - set(train_instance_indexes)))
        user_input_train = user_input[train_instance_indexes]
        item_i_input_train = item_i_input[train_instance_indexes]
        item_j_input_train = item_j_input[train_instance_indexes]
        labels_train = labels[train_instance_indexes]
        user_input_val = user_input[val_instance_indexes]
        item_i_input_val = item_i_input[val_instance_indexes]
        item_j_input_val = item_j_input[val_instance_indexes]

        best_auc_score = 0
        for epoch in range(no_epochs):
            self.model.fit([user_input_train, item_i_input_train, item_j_input_train], [labels_train, labels_train], initial_epoch=epoch, epochs=epoch+1, batch_size=batches, verbose=1, shuffle=True)

            if (epoch % val_interval) == 0:
                user_matrix = self.model.get_layer('UserEmb').get_weights()[0]
                item_matrix = self.model.get_layer('ItemEmb').get_weights()[0]
                auc_scores = []
                for t, (u, i, j) in enumerate(zip(user_input_val, item_i_input_val, item_j_input_val)):
                    auc_scores.append(1 if np.dot(user_matrix[u], item_matrix[i]) > np.dot(user_matrix[u], item_matrix[j]) else 0)
                print('Validation accuracy:', auc_scores.count(1) / len(auc_scores), '(Sample', t, 'of', str(len(val_instance_indexes)) + ')')
                if (auc_scores.count(1) / len(auc_scores)) < best_auc_score:
                    break
                else:
                    best_auc_score = (auc_scores.count(1) / len(auc_scores))

    def predict(self):
        self.predicted_relevance = np.zeros((self.no_users, self.no_items))
        item_pids = np.arange(self.no_items, dtype=np.int32)
        user_matrix = self.model.get_layer('UserEmb').get_weights()[0]
        item_matrix = self.model.get_layer('ItemEmb').get_weights()[0]
        print('Computing predictions')
        for user_id in range(self.no_users):
            user_vector = user_matrix[user_id]
            item_vectors = item_matrix[item_pids]
            self.predicted_relevance[user_id] = np.array(np.dot(user_vector, item_vectors.T))