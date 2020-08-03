
from plant_triplet_net.utils import utils
import os
import pandas as pd
import pickle
import numpy as np
import argparse
import json
import time
import logging


def formatAndAddMetadata(clusters, fuzzy_clusters, metadata):
    metadata['cluster']=clusters
    Nclusters = fuzzy_clusters.shape[1]
    fuzzy_cols = ['fuzzy_cluster_'+('0'+str(c))[-2:] for c in range(Nclusters)]
    metadata[fuzzy_cols] =  pd.DataFrame( fuzzy_clusters, columns=fuzzy_cols)
    return metadata

def main(features_file, clustering_model_file, output_file, log_output_file):
    
    if log_output_file: 
        utils.ensureDir(log_output_file)
        logging.basicConfig(filename=log_output_file, level=logging.INFO, format ='%(asctime)s  - %(levelname)s - %(message)s')
    else: logging.basicConfig(level=logging.INFO)
    logging.info('\n\n-------------- Starting features to clusters --------------\n')
    print('Started features to clusters')
    start_time = time.time()
    
    # Load features and translate relative to t0 
    features, metadata, time_point_feature_size = utils.loadEmbeddings(features_file)
    features = utils.substractFirstTimePoint(features, time_point_feature_size)
    
    # load the TVN clustering model
    model = utils.loadClusteringModel(clustering_model_file)

    # Run model 
    clusters = model.predictClusters(features)
    fuzzy_clusters = model.predictFuzzyClusters(features)

    # format outputs and Save results 
    clusters_df = formatAndAddMetadata(clusters, fuzzy_clusters, metadata)
    clusters_df.to_csv(output_file)
    logging.info('Saved clusters to %s'%output_file)

    t_elapsed = (time.time() - start_time)/60
    logging.info("\n\ntime elapsed: %.1f minutes "% t_elapsed)
    logging.info("\n-------------- Completed features to clusters --------------\n")
    print('Ended features to clusters')
    

if __name__ == '__main__':

    # Get name of the file with all the input settings 
    parser = argparse.ArgumentParser()
    parser.add_argument('-i' ,  help='File with all the inputs' )
    args = parser.parse_args()
    
    # Read json file and
    inputs = json.load(open(args.i))    
    
    # General params
    features_file = inputs['features_file']
    clustering_model_file = inputs['clustering_model_file']
    log_output_file = inputs['log_output_file']
    
    output_file = inputs['output_file']
    utils.ensureDir(output_file)
    if os.path.isfile(output_file): 
        raise OSError('Output file already exists')

    main(features_file, clustering_model_file, output_file, log_output_file)