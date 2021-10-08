# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 19:10:17 2021

@author: youss
"""
import tensorflow.keras.backend as K
import tensorflow.compat.v1 as tf

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def f1_micro(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp_per_class = K.sum(K.cast(y_true*y_pred, 'float'), axis=1)
    tn_per_class = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=1)
    fp_per_class = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=1)
    fn_per_class = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=1)

    p_per_class = tp_per_class / (tp_per_class + fp_per_class + K.epsilon())
    r_per_class = tp_per_class / (tp_per_class + fn_per_class + K.epsilon())

    f1_per_class = 2*p_per_class*r_per_class / (p_per_class+r_per_class+K.epsilon())
    f1_total= K.sum(f1_per_class*K.sum(y_true,axis=1))/ K.sum(y_true)
    
    return f1_total

