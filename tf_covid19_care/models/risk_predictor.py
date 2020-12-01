# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:43:08 2020

@author: wjcongyu
"""

import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM

def SA_Module(inputs, training, name ='SA_Module'):
    '''
    the self-attentino module
    '''
    hidden1 = KL.Dense(512, activation='relu', name=name+'hidden1')(inputs)
   
    t_v = KL.Dense(inputs.get_shape().as_list()[-1] , activation='selu',  use_bias=False,\
                   kernel_initializer=tf.keras.initializers.RandomUniform(minval=1.0, maxval=1.0),\
                   kernel_regularizer = 'l1', name=name+'_t_v')(hidden1)      #, #kernel_regularizer = 'l1', \
    t_softmax = KL.Softmax(name=name+'_t_softmax')(t_v)  
    t_inputs = inputs * t_softmax
    return t_inputs



def build_network(input_shapes, output_size, training, name = 'TreatmentRecommder'):
    '''
    build the network for covid-19 prediction of how long a patient can be cured
    '''
    dtype = tf.float32
    #treatment information
    treatment_info = KL.Input(shape = input_shapes[0], dtype = dtype, name='treatment_info') 
   
    #imaing information: CNN features from CT images
    image_info = KL.Input(shape = input_shapes[1]+[1], dtype = dtype, name='image_info')   
    base_filters = 16    
    x11 = KL.Conv3D(base_filters, (3, 3, 3), activation='relu', padding='same', name = 'x11')(image_info)  
    x12 = KL.Conv3D(base_filters, (3, 3, 3), activation='relu', padding='same', name = 'x12')(x11)  
    x13 = KL.Conv3D(base_filters, (3, 3, 3), activation='relu', padding='same', name = 'x13')(x12) 
    
    d1 = KL.MaxPool3D()(x13)
    
    x21 = KL.Conv3D(base_filters*2, (3, 3, 3), activation='relu', padding='same', name = 'x21')(d1)  
    x22 = KL.Conv3D(base_filters*2, (3, 3, 3), activation='relu', padding='same', name = 'x22')(x21)  
    
    d2 = KL.MaxPool3D()(x22)
    
    x31 = KL.Conv3D(base_filters*4, (3, 3, 3), activation='relu', padding='same', name = 'x31')(d2)  
    x32 = KL.Conv3D(base_filters*4, (3, 3, 3), activation='relu', padding='same', name = 'x32')(x31)  
   
    d3 = KL.MaxPool3D()(x32)
    
    x41 = KL.Conv3D(base_filters*8, (3, 3, 3), activation='relu', padding='same', name = 'x41')(d3)  
    x42 = KL.Conv3D(base_filters*8, (3, 3, 3), activation='relu', padding='same', name = 'x42')(x41)  
  
    d4 = KL.MaxPool3D()(x42)
    
    x51 = KL.Conv3D(base_filters*16, (3, 3, 3), activation='relu', padding='same', name = 'x51')(d4)  
    x52 = KL.Conv3D(base_filters*16, (3, 3, 3), activation='relu', padding='same', name = 'x52')(x51)  
 
    d5 = KL.MaxPool3D()(x52)
    cnn_GAP = KL.GlobalAveragePooling3D(name='CNN_GAP')(d5)
    cnn_cof = KL.Dense(1, activation='relu', name='cnn_cof')(cnn_GAP)
    
    #patient information
    patient_info = KL.Input(shape = input_shapes[2], dtype = dtype, name='patient_info')
    pcnn_info = KL.Concatenate()([patient_info, cnn_cof])    
    
    #cured probability distruibution subnetwork
    w_pcnn_info = SA_Module(pcnn_info, training)
    
    fc1 = KL.Dense(256, activation='relu', name='fc1')(KL.Concatenate()([w_pcnn_info, cnn_GAP, treatment_info])) 
    fc2 = KL.Dense(512, activation='relu', name='fc2')(fc1) 
    fc3 = KL.Dense(512, activation='relu', name='fc3')(fc2) 
   
    fc_cls = KL.Dense(256, activation='relu', name='fc_cls')(fc3) 
    fc_cls = KL.Dropout(0.4)(fc_cls, training = training)
    severity_cls_preds = KL.Dense(output_size[0],activation='softmax', name='severity_cls_preds')(fc_cls)
    
    fc_reg = KL.Dense(256, activation='relu', name='fc_reg')(fc3)
    fc_reg = KL.Dropout(0.4)(fc_reg, training = training)
    risk_reg_preds = KL.Dense(output_size[1],activation='softmax', name='risk_reg_preds')(fc_reg)
    
    model = KM.Model([treatment_info,image_info,patient_info], [severity_cls_preds, risk_reg_preds], name=name)
    return model
    
