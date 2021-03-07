#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os

from helpers.instances_creator import generator
from helpers.utils import load_obj, save_obj
from models.model import Model

class PointWise(Model):

    def __init__(self, users, items, observed_relevance, unobserved_relevance, category_per_item, item_field, user_field, rating_field):
        super().__init__(users, items, observed_relevance, unobserved_relevance, category_per_item, item_field, user_field, rating_field)

    def __get_model(self, mf_dim=10, layers=np.array([10])):
        no_layer = len(layers)

        user_input = tf.keras.layers.Input(shape=(1,), dtype='int32', name='user_input')
        item_input = tf.keras.layers.Input(shape=(1,), dtype='int32', name='item_input')

        MF_Embedding_User = tf.keras.layers.Embedding(input_dim=self.no_users, output_dim=mf_dim, name='mf_embedding_user', input_length=1)
        MF_Embedding_Item = tf.keras.layers.Embedding(input_dim=self.no_items, output_dim=mf_dim, name='mf_embedding_item', input_length=1)

        MLP_Embedding_User = tf.keras.layers.Embedding(input_dim=self.no_users, output_dim=int(layers[0]/2), name='mlp_embedding_user', input_length=1)
        MLP_Embedding_Item = tf.keras.layers.Embedding(input_dim=self.no_items, output_dim=int(layers[0]/2), name='mlp_embedding_item', input_length=1)

        mf_user_latent = tf.keras.layers.Flatten()(MF_Embedding_User(user_input))
        mf_item_latent = tf.keras.layers.Flatten()(MF_Embedding_Item(item_input))
        mf_vector = tf.keras.layers.Multiply()([mf_user_latent, mf_item_latent])

        mlp_user_latent = tf.keras.layers.Flatten()(MLP_Embedding_User(user_input))
        mlp_item_latent = tf.keras.layers.Flatten()(MLP_Embedding_Item(item_input))
        mlp_vector = tf.keras.layers.Concatenate()([mlp_user_latent, mlp_item_latent])
        for idx in range(1, no_layer):
            layer = tf.keras.layers.Dense(layers[idx], activation='relu', name="layer%d" % idx)
            mlp_vector = layer(mlp_vector)

        predict_vector = tf.keras.layers.Concatenate()([mf_vector, mlp_vector])
        prediction = tf.keras.layers.Dense(1, activation='sigmoid', name="prediction")(predict_vector)
        model = tf.keras.models.Model(inputs=[user_input, item_input], outputs=[prediction])

        return model

    def train(self, no_epochs=20, batches=1024, lr=0.001, no_factors=10, no_negatives=10, gen_mode='point', val_split=0.1):
        print('Generating training instances', 'of type', gen_mode)
        x, y = generator(self.observed_relevance, self.categories, self.no_categories, self.category_per_item, self.categories_per_user, no_negatives=no_negatives, gen_mode=gen_mode)

        print('Performing training -', 'Epochs', no_epochs, 'Batch Size', batches, 'Learning Rate', lr, 'Factors', no_factors, 'Negatives', no_negatives, 'Mode', gen_mode)
        self.model = self.__get_model(no_factors, np.array([64, 32, 16, 8], np.int32))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss='binary_crossentropy')

        user_input, item_i_input, _ = x
        labels = y
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1)]
        self.model.fit([np.array(user_input), np.array(item_i_input)], np.array(labels), validation_split=val_split, batch_size=batches, epochs=no_epochs, verbose=1, shuffle=True, callbacks=callbacks)

    def predict(self):
        self.predicted_relevance = np.zeros((self.no_users, self.no_items))
        for user_id in self.users:
            if (user_id % 100) == 0:
                print('\rComputing predictions for user', user_id, '/', self.no_users, end='')
            user_data = np.array((np.ones(self.no_items)*user_id).tolist())
            item_data = np.array(self.items)
            self.predicted_relevance[user_id] = np.squeeze(self.model.predict([user_data, item_data]).tolist())