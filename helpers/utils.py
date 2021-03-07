#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pickle

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def compute_gini(x, w=None):
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def compute_weights(item_id, tail_head, tail_long, mrtype, list_u, p_d_u):
    psum = 0
    for tset_pos, tset in enumerate([tail_head, tail_long]):
        p_v_d = (1 if item_id in tset else 0)
        ies = ((np.prod([1 - (1 if ranked_item_id in tset else 0) for ranked_item_id in list_u])) if mrtype == 'balanced' else (1 - len(list(set(tset) & set(list_u))) / len(list_u))) if len(list_u) > 1 else 1
        psum += p_d_u[tset_pos] * p_v_d * ies
    return psum

def get_bpr_loss(y_true, y_pred):
    return 1.0 - tf.keras.backend.sigmoid(y_pred)

def get_dot_difference_shape(shapeVectorList):
    userEmbeddingShapeVector, itemPositiveEmbeddingShapeVector, itemNegativeEmbeddingShapeVector = shapeVectorList
    return userEmbeddingShapeVector[0], 1

def get_dot_difference(parameterMatrixList):
    userEmbeddingMatrix, itemPositiveEmbeddingMatrix, itemNegativeEmbeddingMatrix = parameterMatrixList
    return tf.keras.backend.batch_dot(userEmbeddingMatrix, itemPositiveEmbeddingMatrix, axes=1) - tf.keras.backend.batch_dot(userEmbeddingMatrix, itemNegativeEmbeddingMatrix, axes=1)

def get_correlation_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = tf.keras.backend.mean(x)
    my = tf.keras.backend.mean(y)
    xm, ym = x-mx, y-my
    r_num = tf.keras.backend.sum(tf.multiply(xm,ym))
    r_den = tf.keras.backend.sqrt(tf.multiply(tf.keras.backend.sum(tf.keras.backend.square(xm)), tf.keras.backend.sum(tf.keras.backend.square(ym))))
    r = r_num / tf.where(tf.equal(r_den, 0), 1e-3, r_den)
    r = tf.keras.backend.abs(tf.keras.backend.maximum(tf.keras.backend.minimum(r, 1.0), -1.0))
    return tf.keras.backend.square(r)
