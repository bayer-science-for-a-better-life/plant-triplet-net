import torch.optim as optim
from torch.optim import lr_scheduler
from itertools import cycle
import time
import copy
import logging

def trainEpoch( model, optimizer, phase, dataloaders, loss_hist ):
    
    running_loss = 0.0

    for batch_count, data in enumerate(dataloaders[phase]):

        # Run model (Forward)
        image_batch = data.to(model.device) 
        optimizer.zero_grad()  
        features_t_stack, _ = model(image_batch)

        # Calculate losses
        loss = model.getTimeCourseRankingLoss(features_t_stack)

        # Update parameters (backward) 
        if phase == 'train':
            loss.backward()
            optimizer.step()

        # Report an keep track of results
        running_loss += loss.item() * image_batch.shape[0]
        loss_hist['minibatch'][phase].append(loss.item())
        if ((batch_count%50)==0):
            logging.info('%s batch %i, loss: %.5f'%(phase, batch_count, loss.item()))
    
    epoch_loss = running_loss / len(dataloaders[phase].dataset)
    loss_hist['epoch'][phase].append(epoch_loss)
    logging.info('%s epoch loss: %.10f '%(phase, epoch_loss ))

    return epoch_loss

def extendLossHistory(loss_hist_1,loss_hist_2):
    for k1 in loss_hist_1.keys():
        for k2 in loss_hist_1[k1].keys():
            if isinstance(loss_hist_1[k1][k2], dict): 
                for k3 in loss_hist_1[k1][k2].keys():
                    loss_hist_1[k1][k2][k3]= loss_hist_1[k1][k2][k3] + loss_hist_2[k1][k2][k3]
            else: 
                loss_hist_1[k1][k2]= loss_hist_1[k1][k2] + loss_hist_2[k1][k2]
    return loss_hist_1

def trainModel( model, dataloaders, train_params):
    
    """     
    :param model: a Pytorch neural net model derived from nn.Module
    :param dataloaders: dictionary of Pytorch Dataloader objects feeding time course images for the training (dataloaders['train']) and validation sets (dataloaders['val'])
    :param train_params: all optimization and regularization parameters
        - regarding network regularization: 
            train_params['weight_decay']: L2 penalty on the weigths
        - regarding optimization objective:
            train_params['ranking_gap']: gap parameter in the ranking loss
            train_params['distance_type']: used in ranking loss, either eculidian ('euc') or 1 - cosinesimilarity ('cos')
        - regarding optimizer:
        train_params['Nepochs']: Number of times the unlabeled training set will be iterated (serving the ranking loss).
            labeled datasets in the semi-supervised setting are repeted in a loop to fit the unsupervised epochs
        train_params['optimizer_type']: either 'Adam' or 'SGD' supported, but Adam is highly recomended
        train_params['learning_rate']: optimizer initial learning rate
    """
    
    # Create optimizer
    if train_params['optimizer_type'] =='SGD':
        optimizer = optim.SGD( model.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'], momentum=0.9)
    elif train_params['optimizer_type'] == 'Adam':
        optimizer = optim.Adam( model.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict()) 
    best_val_loss = 1E10
    loss_hist =  { 'epoch'    :{'train':[], 'val':[] } , 
                   'minibatch':{'train':[], 
                                  'val':[]}} 
    
    for epoch in range(train_params['Nepochs']):
        logging.info('\nEpoch {}/{} \n ------------'.format(epoch, train_params['Nepochs'] - 1))
        
        # Each epoch has a training and validation phase
        if epoch ==0:
            phases = ['val','train','val'] # Compute validation loss before starting training
        else:
            phases = ['train','val']
        for phase in phases:
            
            # Set model to training or eval mode: 
            # trainign mode: includes dropout and updates running stats of batch normalization layers.
            # eval mode: no droput or updates of running stats. 
            if phase == 'train':
                scheduler.step() 
                model.train() 
                if model.raw_feature_extractor.trainable==False:
                    model.raw_feature_extractor.eval()  # The transfer learning part stays in eval mode 
            else:
                model.eval()
                
            # train epoch
            epoch_loss = trainEpoch( model, optimizer, phase, dataloaders, loss_hist )
            
            # Deep copy the model if it has the best validation accuracy
            if (phase == 'val') & (epoch_loss < best_val_loss):
                best_val_loss = epoch_loss
                best_val_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    # report final results
    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val loss: %.4f at epoch %i'%(best_val_loss, best_val_epoch))
    
    return model, loss_hist

