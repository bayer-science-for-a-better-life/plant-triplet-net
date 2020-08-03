from plant_triplet_net.utils import model_zoo as mzoo
from plant_triplet_net.utils import dataloaders as dload
from plant_triplet_net.utils import train
from plant_triplet_net.utils import utils
import os
import pandas as pd
import argparse
import json
import time
import logging

def main( experiment_file, dataloader_info, architecture, train_params, device, output_file, log_output_file ):
    
    if log_output_file: 
        utils.ensureDir(log_output_file)
        logging.basicConfig(filename=log_output_file, level=logging.INFO, format ='%(asctime)s  - %(levelname)s - %(message)s')
    else: logging.basicConfig(level=logging.INFO)
    logging.info('\n\n-------------- Starting training feature extraction model --------------\n')
    print('Started training feature extraction model')
    start_time = time.time()

    # Read and check experiment file 
    experiment_df = pd.read_csv(experiment_file, index_col=0)
    experiment_df = utils.discardIncompleteTimeCourses(experiment_df, dataloader_info['time_points'], dataloader_info['crash_incomplete_time_courses'] )
    if len(experiment_df)==0:
        raise ValueError('Error in train_feature_extraction_model.py. Experiment is empty after removing incomplete time courses. Double check that you selected the correct time_points in the arguments json file')
    
    # Split data into train and val and create respective dataloaders
    datasets_info = dict()
    datasets_info['train'] = experiment_df.loc[ experiment_df['set_type']=='train' ].reset_index(drop="index")
    datasets_info['val']= experiment_df.loc[ experiment_df['set_type']=='val' ].reset_index(drop="index")
    dataloaders, _ = dload.createDataLoaders(datasets_info, dataloader_info['time_points'], dataloader_info['batch_size'], architecture['image_size'] )

    # Instantiate model
    model = mzoo.TimeCourseTripletNet( architecture, train_params=train_params )
    model.to(device)
    logging.info(model)

    # step 1: transfer learning module with frozen parameters
    logging.info('\n\nStep 1. Training ranking module alone (transfer learning frozen):\n')        
    model.raw_feature_extractor.freezeParams()
    if train_params['Nepochs']['step_1']>0:
        train_params_1 = train_params.copy()
        train_params_1['Nepochs'] = train_params_1['Nepochs']['step_1']
        train_params_1['learning_rate'] = train_params_1['learning_rate']['step_1']
        model.trainableParametersSummary()
        model, loss_hist_1 = train.trainModel( model, dataloaders, train_params_1)

    # step 2: transfer learning module with trainable parameters
    if train_params['Nepochs']['step_2']>0:
        logging.info('\n\nStep 2. Training ranker and transfer learning modules:\n')
        train_params_2 = train_params.copy()
        train_params_2['Nepochs'] = train_params_2['Nepochs']['step_2']
        train_params_2['learning_rate'] = train_params_2['learning_rate']['step_2']
        model.raw_feature_extractor.unfreezeParams()
        model.trainableParametersSummary()
        model, loss_hist_2 = train.trainModel( model, dataloaders, train_params_2)

    # merge loss histroy from btoh steps and plot training
    if  (train_params['Nepochs']['step_1']>0) & (train_params['Nepochs']['step_2']>0):
        loss_hist = train.extendLossHistory(loss_hist_1,loss_hist_2)
    elif train_params['Nepochs']['step_1']>0:
        loss_hist = loss_hist_1
    elif train_params['Nepochs']['step_2']>0:
        loss_hist = loss_hist_2
    
    # Save model together with all the training info and history. We could also add user and date
    utils.saveFeatureExtractionModel(model,architecture,train_params,loss_hist,dataloader_info,output_file)
    
    t_elapsed = (time.time() - start_time)/60
    logging.info("\n\ntime elapsed: %.1f minutes "% t_elapsed)
    logging.info("\n-------------- Completed training --------------\n")
    print('Ended training')
    
if __name__ == '__main__':

    # Get name of the file with all the input settings 
    parser = argparse.ArgumentParser()
    parser.add_argument('-i' ,  help='File with all the inputs' )
    args = parser.parse_args()
    
    # Read json file and
    inputs = json.load(open(args.i))    
    
    # General params
    experiment_file = inputs['experiment_file']
    device = inputs['device']
    log_output_file = inputs['log_output_file']
    output_file = inputs['output_file']
    utils.ensureDir(output_file)
    if os.path.isfile(output_file): 
        raise OSError('Output file already exists')

    # Dataloaders 
    dataloader_info = inputs['dataloader_info']
    required_keys = ['time_points','batch_size','crash_incomplete_time_courses']
    utils.checkKeys(dataloader_info, required_keys)
    
    # Model architecture
    architecture = inputs['architecture']
    architecture['Nt'] = len(dataloader_info['time_points'])  
    required_keys = ['raw_feature_extractor','ranker','dropout','image_size','embedding_size','Nt']
    utils.checkKeys(architecture, required_keys)

    # Training parameters
    train_params = inputs['train_params']
    required_keys= ['Nepochs', 'ranking_gap', 'distance_type','optimizer_type', 'learning_rate', 'weight_decay']
    utils.checkKeys(train_params, required_keys)
    utils.checkKeys( train_params['Nepochs'], ['step_1','step_2'])
    
    main(experiment_file, dataloader_info, architecture,train_params, device, output_file, log_output_file)