# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:41:30 2020

@author: wjcongyu
"""
import _init_pathes
import os
import tensorflow as tf
from configs.cfgs import cfg
from models.risk_predictor import build_network
from data_process.data_generator import DataGenerator 
from data_process.data_processor import readCsv 
from models.backend import find_weights_of_last
import numpy as np
import argparse
import os.path as osp
from tensorflow.keras import models as KM
import pandas as pd
import datetime
parser = argparse.ArgumentParser()
parser.add_argument('-train_subsets', '--train_subsets', help='the subsets for training.', type = str, default ='1;2;3')
parser.add_argument('-eval_subsets', '--eval_subsets', help='the subset for test, others for training.', type = str, default ='4.lbl')
parser.add_argument('-batch_size', '--batch_size', help='the mini-batch size.', type = int, default = 72)
parser.add_argument('-cuda_device', '--cuda_device', help='runining on specified gpu', type = int, default = 0)
parser.add_argument('-gt_treatment', '--gt_treatment', help='using treatment or not.', type = int, default = 0)#1:yes 0:no
parser.add_argument('-save_root', '--save_root', help='root path to save the prediction results.', type = str, default = 'eval_results')#1:yes 0:no
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
   
    eval_files = []
    eval_subsets = args.eval_subsets.split(';')
    for i in range(len(eval_subsets)):
        eval_files.append(os.path.join(cfg.data_set, eval_subsets[i]))
   
    val_data_generator = DataGenerator(eval_files, cfg, train_mode=False)
    eval_sample_num = val_data_generator.load_dataset()
   
    treattype = {0:'WithoutTreatment', 1:'WithTreatment'}
    model_dir_name = args.train_subsets +'_modelweights_'+ treattype[args.gt_treatment]
   
    model = build_network([ cfg.treatment_infosize, cfg.im_feedsize, cfg.patient_infosize], [cfg.severity_categories, cfg.time_range], False, name='risk_predictor')

    feature_idx = [23, 34, 35]
    
    checkpoint_file = find_weights_of_last(os.path.join(cfg.CHECKPOINTS_ROOT, model_dir_name), 'risk_predictor')
    #print('############################',os.path.join(cfg.CHECKPOINTS_ROOT, model_dir_name))
    if checkpoint_file != '':  
        print ('@@@@@@@@@@ loading pretrained from ', checkpoint_file)
        model.load_weights(checkpoint_file)
    else:
        assert('no weight file found!!!')
   
    print (model.summary())
    
    #print layer information
    for i in range(len(model.layers)):
        layer = model.layers[i]      
        print(i, layer.name, layer.output.shape)
           
    #define the output of the network to get 
    outputs = [model.layers[i].output for i in feature_idx]
    pred_model = KM.Model(inputs=model.inputs, outputs=outputs)    
   
    save_root = args.save_root
    if not osp.exists(save_root):
        os.mkdir(save_root)
        
    rst_save_root = osp.join(save_root, treattype[args.gt_treatment])
    if not osp.exists(rst_save_root):
        os.mkdir(rst_save_root)
        
    feature_significance = []   
    severity_cls_preds = []
    risk_reg_preds = []
    gt_hitday = []
    gt_eventindicator = []
    gt_features = []
    gt_covid_severity = []
   
    for step in range(eval_sample_num//(args.batch_size)+1):
        start = datetime.datetime.now()
        evbt_painfo, evbt_treatinfo, evbt_ims, evbt_treattimes,evbt_censor_indicator, evbt_severity  \
                                                = val_data_generator.next_batch(args.batch_size)
        if args.gt_treatment==0:
            feed_treatinfo= tf.zeros_like(evbt_treatinfo)
        else:
            feed_treatinfo = evbt_treatinfo
       
        feed = [feed_treatinfo, evbt_ims, evbt_painfo]
            
        coff, cls_pred, reg_pred = pred_model(feed, training=False)
        end = datetime.datetime.now()
        print('processing time:', end-start)      
       
        severity_cls_preds.append(cls_pred)
        risk_reg_preds.append(reg_pred)
        feature_significance.append(coff)
        gt_hitday.append(evbt_treattimes)
        gt_eventindicator.append(evbt_censor_indicator)
        gt_features.append(evbt_painfo)
        gt_covid_severity.append(evbt_severity)
       
            
    severity_cls_preds = np.concatenate(severity_cls_preds, axis=0)
    risk_reg_preds = np.concatenate(risk_reg_preds, axis=0)
    feature_significance = np.concatenate(feature_significance, axis=0)
    gt_hitday = np.concatenate(gt_hitday, axis=0)
    gt_eventindicator = np.concatenate(gt_eventindicator, axis=0)
    gt_features = np.concatenate(gt_features, axis=0)
    gt_covid_severity = np.concatenate(gt_covid_severity, axis=0)
    
    
    pinfo_header = readCsv(eval_files[0])[0][1:50]                
    pinfo_header = pinfo_header[0:2]+pinfo_header[3:]    
    
    csv_file = os.path.join(rst_save_root, '{0}_risk_reg_preds.csv'.format(args.eval_subsets))
    save_data = pd.DataFrame(risk_reg_preds, columns=['day '+str(i+1) for i in range(cfg.time_range)])
    save_data.to_csv(csv_file,header=True, index=False)
    
    csv_file = os.path.join(rst_save_root, '{0}_severity_cls_preds.csv'.format(args.eval_subsets))
    save_data = pd.DataFrame(severity_cls_preds, columns=['low-risk(mild&moderate)','high-risk(severe&critical)'])
    save_data.to_csv(csv_file,header=True, index=False)
    
    csv_file = os.path.join(rst_save_root, '{0}_gt_hitday.csv'.format(args.eval_subsets))
    save_data = pd.DataFrame(gt_hitday, columns=['hit day'])
    save_data.to_csv(csv_file,header=True, index=False)
    
    csv_file = os.path.join(rst_save_root, '{0}_indicator.csv'.format(args.eval_subsets))
    save_data = pd.DataFrame(gt_eventindicator, columns=['indicator'])
    save_data.to_csv(csv_file,header=True, index=False)
    
    csv_file = os.path.join(rst_save_root, '{0}_clinic_features.csv'.format(args.eval_subsets))
    save_data = pd.DataFrame(gt_features, columns=pinfo_header)
    save_data.to_csv(csv_file,header=True, index=False)
   
    csv_file = os.path.join(rst_save_root, '{0}_gt_severity.csv'.format(args.eval_subsets))
    save_data = pd.DataFrame(gt_covid_severity, columns=['severity'])
    save_data.to_csv(csv_file,header=True, index=False)  
   
       
    pinfo_header.append('cnn_feature')
    csv_file = os.path.join(rst_save_root, '{0}_feature_significance.csv'.format(args.eval_subsets))
    save_data = pd.DataFrame(feature_significance, columns=pinfo_header)
    save_data.to_csv(csv_file,header=True, index=False)
    
  

if __name__ == "__main__":
    args = parser.parse_args()
    print (args)
    
    main(args)
