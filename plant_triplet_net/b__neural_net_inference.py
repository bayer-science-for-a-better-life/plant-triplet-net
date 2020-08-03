
from plant_triplet_net.utils import dataloaders as dload
from plant_triplet_net.utils import utils
from plant_triplet_net.utils import visualization as dv
import os
import pandas as pd
import numpy as np
import torch
import argparse
import json
import time
import logging

def evaluateModel(model, dataloader, Nsamples):
    embedding_size  = model.architecture['embedding_size']*model.architecture['Nt']
    embeddings  = np.zeros((Nsamples,embedding_size), dtype=np.float32)
    c = 0
    with torch.no_grad():
        for image_batch in dataloader:
            image_batch = image_batch.to(device)
            batch_size = image_batch.shape[0]
            _, batch_embedding = model(image_batch)
            embeddings[c:c+batch_size,:] =  batch_embedding.cpu().numpy()
            c = c + image_batch.shape[0]
            if ((c/batch_size)%100)==0:
                percent = float(c)/Nsamples*100
                logging.info('%.1f percent procesed'%percent)
    return embeddings

def formatAndAddMetadata(embeddings, metadata_df, time_points):
    embedding_cols = []
    assert (embeddings.shape[1]%len(time_points) == 0)
    Nf = int(embeddings.shape[1]/len(time_points)) # num fetures per time point
    for t in time_points:
        embedding_cols.extend([ 'feature_'+ ('000'+str(f))[-4:] + '_t'+str(t) for f in range(Nf) ])
    metadata_df[embedding_cols]= pd.DataFrame( embeddings, columns=embedding_cols)
    return metadata_df

def main( model_file, image_paths_file, dataloader_info, device, output_file, log_output_file ):
    
    if log_output_file: 
        utils.ensureDir(log_output_file)
        logging.basicConfig(filename=log_output_file, level=logging.INFO, format ='%(asctime)s  - %(levelname)s - %(message)s')
    else: logging.basicConfig(level=logging.INFO)
    logging.info('\n\n-------------- Starting feature extraction --------------\n')
    print('Started feature extraction')
    start_time = time.time()
    
    # Load model and set it to EVAL mode
    model, model_info = utils.loadFeatureExtractionModel(model_file)
    model.eval()
    model.to(device)
    
    # Set the number of time points in the model to the time points you want to evaluate. Remember that the model can be trained and evaluated on different time points
    time_points =  dataloader_info['time_points']
    model.architecture['Nt'] = len(time_points)

    # read image_paths_file,  select only desired time points 
    paths_df = pd.read_csv(image_paths_file, index_col=0)
    paths_df = utils.discardIncompleteTimeCourses(paths_df,time_points, dataloader_info['crash_incomplete_time_courses'])
    if len(paths_df)==0:
        raise ValueError('Error in image_to_features.py. list of image paths is empty after removing incomplete time courses. Double check that you selected the correct time_points in the arguments json file')
        
    # Create dataloaders
    datasets_info = {'eval':paths_df }
    dataloaders, datasets = dload.createDataLoaders(datasets_info, time_points, dataloader_info['batch_size'], model.architecture['image_size'])
    dataloader, dataset  = dataloaders['eval'], datasets['eval']
    Nsamples = int( len(paths_df)/len(time_points) )

    # Run model and save results
    embeddings = evaluateModel(model,dataloader, Nsamples)
    output_df = formatAndAddMetadata(embeddings, dataset.groups_metadata , time_points )
    output_df.to_csv(output_file)
    logging.info('Saved Embeddings to %s'%output_file)
        
    t_elapsed = (time.time() - start_time)/60
    logging.info("\n\ntime elapsed: %.1f minutes "% t_elapsed)
    logging.info("\n-------------- Ended feature extraction --------------\n")
    print('Ended feature extraction')

    
if __name__ == '__main__':

    # Get name of the file with all the input settings 
    parser = argparse.ArgumentParser()
    parser.add_argument('-i' ,  help='File with all the inputs' )
    args = parser.parse_args()
    
    # Read json file and
    inputs = json.load(open(args.i))    
    
    # General params
    model_file = inputs['feature_extraction_model']
    image_paths_file = inputs['image_paths_file']
    device = inputs['device']
    output_file = inputs['output_file']
    log_output_file = inputs['log_output_file']
    utils.ensureDir(output_file)
    if os.path.isfile(output_file): 
        raise OSError('Output file already exists')

    # Dataloaders 
    dataloader_info = inputs['dataloader_info']
    required_keys = ['time_points','batch_size','crash_incomplete_time_courses']
    utils.checkKeys(dataloader_info, required_keys)

    main(model_file, image_paths_file, dataloader_info, device, output_file , log_output_file)
