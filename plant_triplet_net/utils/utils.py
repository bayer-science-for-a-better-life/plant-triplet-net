from plant_triplet_net.utils import model_zoo as mzoo
import os
import pickle
import pandas as pd
import numpy as np
from importlib import import_module
import torch
import logging

def checkKeys(input_dict, required):
    missing_keys = [key for key in  required if ( key in input_dict )==False ]
    if missing_keys:
        raise KeyError('Missing keys in input json file: ', missing_keys)

def ensureDir(file_path_or_dir):
    if '.' in file_path_or_dir:
        directory = os.path.dirname(file_path_or_dir)
    else:
        directory = file_path_or_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def discardIncompleteTimeCourses(experiment_df, time_points, raise_error ):
    experiment_df = experiment_df.loc[ experiment_df['time_point'].isin(time_points)]  
    old_l = len(experiment_df)
    experiment_df = experiment_df.groupby(['experiment_ID','plate_ID','well_ID']).filter(lambda x: len(x)==len(time_points) )
    Nincomplete = (old_l -len(experiment_df))
    if Nincomplete>0:
        if raise_error:
            logging.error('%i files with incomplete time-courses'%Nincomplete) 
            raise ValueError('Experiment file has incomplete time-courses')
        else:
            logging.warning('Removed %i files with incomplete time-courses'%Nincomplete ) 
    return experiment_df

def loadFeatureExtractionModel(model_file, only_info=False):
    model_info = torch.load(model_file)
    if not only_info:
        model = mzoo.TimeCourseTripletNet( model_info['architecture'] )
        model.load_state_dict(model_info['state_dict'])
        logging.info('Loaded feature extraction model from %s'%model_file)
        return model, model_info
    else:
        return model_info

def saveFeatureExtractionModel(model, architecture, train_params, loss_hist, dataloader_info, output_file):
    torch.save({'architecture':architecture, 'train_params':train_params, 'dataloader_info':dataloader_info,
                 'state_dict': model.state_dict(),'loss_hist': loss_hist, 
                 'raw_feature_extractor_class':type(model.raw_feature_extractor).__name__} , output_file)
    logging.info('Saved feature extraction model to %s'%output_file)
    
def loadClusteringModel(model_file):
    with open(model_file,'rb') as f:
        model = pickle.load(f)
        logging.info('Loaded TVN clustering model from %s'%model_file)
    return model

def saveClusteringModel(model, output_file):
    with open(output_file, 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
    logging.info('saved TVN clustering model to %s'%output_file)
    
def loadEmbeddings(embeddings_file):
    
    # Load features file 
    results = pd.read_csv(embeddings_file, index_col =0, low_memory=False)

    # Split features and metadata cols and get features as an np array
    feature_cols = [col for col in results.columns if 'feature' in col]
    if len(feature_cols)==0:
        logging.error('File %s does not contain any features'%embeddings_file)
        raise ValueError('File %s does not contain any features'%embeddings_file)
    metadata_cols = [col for col in results.columns if not('feature' in col)]
    features = results[feature_cols].values.astype(np.float32)
    metadata = results[metadata_cols]
    
    # Infer the number of features per time point
    time_point_features = np.unique( [f[0:f.find('_t')] for f in feature_cols] )
    Nfeatures = len(time_point_features) # feature size for a single time point
    
    return features, metadata, Nfeatures

def substractFirstTimePoint( features, time_point_feature_size):
    """
    # Translate features relative to the first time point. This is important to represent phentoypes in 
    # terms of changes with respect to initial state of the plant. 
    # It also reduces the dimensionality of the time-course features sicne the first time point is dropped (only zeros)
    """
    Nsamples = features.shape[0]
    features_tstack = features.reshape(Nsamples ,-1, time_point_feature_size  )
    features_delta  = features_tstack[:,1:,:]-features_tstack[:,0:1,:]
    features_delta  = features_delta.reshape(Nsamples ,-1 )
    return features_delta

