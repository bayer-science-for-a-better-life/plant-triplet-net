from plant_triplet_net.utils import model_zoo as mzoo
from plant_triplet_net.utils import utils
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.cluster import KMeans
from scipy import stats
import argparse
import json
import time
import logging
random_seed = 37

def main(features_file, kmeans_params, tvn_params, output_file , log_output_file):

    if log_output_file: 
        utils.ensureDir(log_output_file)
        logging.basicConfig(filename=log_output_file, level=logging.INFO, format ='%(asctime)s  - %(levelname)s - %(message)s')
    else: logging.basicConfig(level=logging.INFO)
    logging.info('\n\n-------------- Starting training TVN clustering model --------------\n')
    print('Started training TVN clustering model')
    start_time = time.time()
    
    # Load embeddings file and extract features in a numpy array. Although not mandatory, it is recommended to substract all features minus the respective feature at time-point 0. This allows to express time-course embeddings in terms of relative changes with respect to the inital state
    features, metadata, time_point_feature_size = utils.loadEmbeddings(features_file)
    features = utils.substractFirstTimePoint(features, time_point_feature_size)
    
    # Create a features scaler based on Typical Variance normalization trained on controls and normalize features
    tvn_scaler = mzoo.TVNScaler(tvn_params['epsilon_ratio'])
    ctrl_idx = (metadata['set_type']=='ctrl').values 
    ctrl_features = features[ctrl_idx,:]
    tvn_scaler.fit(ctrl_features)
    features_norm = tvn_scaler.transform(features)
    
    # Select a random subset of controls to include in the clustering process. 
    # In principle we will apply kmeans on treated samples. However, to asure that we include healthy samples, we take a small proportion of controls idx_treated
    idx_treated = metadata.loc[ ctrl_idx==False ].index.values
    Nctrl = int(len(idx_treated)*kmeans_params['percent_ctrl_downsample'])
    down_sample_idx_ctrl = metadata.loc[ ctrl_idx ].sample(Nctrl,random_state=random_seed).index.values
    down_sample_idx = np.concatenate( [idx_treated, down_sample_idx_ctrl])
    features_norm_down = features_norm[down_sample_idx,:]

    # fit kmeans model 
    kmeans_model = KMeans(n_clusters=kmeans_params['Nclusters'], n_init=kmeans_params['Ninit'], tol=1e-6, random_state=random_seed)
    kmeans_model.fit(features_norm_down)

    # Run the model on the full dataset, identifiy the cluster with most controls
    clusters = kmeans_model.predict( features_norm )            
    ctrl_cluster = stats.mode( clusters[ metadata[ctrl_idx].index ] )[0][0]
    
    # Create a TVNClusterer object which puts tvn_scaler and clustering together and sort clusters by increasing distance to the predominant control's cluster
    model = mzoo.TVNClusterer( tvn_scaler, kmeans_model ) 
    model.sortCentroids( ctrl_cluster )

    # Save model
    utils.saveClusteringModel(model,output_file)
    
    t_elapsed = (time.time() - start_time)/60
    logging.info("\n\ntime elapsed: %.1f minutes "% t_elapsed)
    logging.info("\n-------------- Completed training TVN clustering model --------------\n")
    print('Ended training')
    
if __name__ == '__main__':

    # Get name of the file with all the input settings 
    parser = argparse.ArgumentParser()
    parser.add_argument('-i' ,  help='File with all the inputs' )
    args = parser.parse_args()
    
    # Read json file and
    inputs = json.load(open(args.i))    
    
    # General params
    features_file = inputs['features_file']
    
    kmeans_params = inputs['kmeans']
    required_keys = ['Nclusters','Ninit','percent_ctrl_downsample']
    utils.checkKeys(kmeans_params, required_keys)

    tvn_params = inputs['tvn']
    required_keys = ['epsilon_ratio']
    utils.checkKeys(tvn_params, required_keys)
    
    log_output_file = inputs['log_output_file']
    output_file = inputs['output_file']
    utils.ensureDir(output_file)
    if os.path.isfile(output_file): 
        raise OSError('Output file already exists')

    main(features_file, kmeans_params, tvn_params, output_file , log_output_file)