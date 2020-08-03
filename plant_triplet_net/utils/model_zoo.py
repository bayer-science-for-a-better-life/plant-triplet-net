import os
import numpy as np
import pandas as pd

# for feature extraciton models
import torch
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from random import shuffle as randomshuffle

# For TVN and clustering models
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import spatial
import logging 
       
class TimeCourseTripletNet(nn.Module):
    
    """
    Model for self-supervised feature extraction from time-course images trained on a triplets ranking loss
    """
    
    def __init__(self, architecture, train_params = {}, device = torch.device("cpu") ):
        super(TimeCourseTripletNet, self).__init__()
            
        self.architecture  = architecture
        self.train_params = train_params
        self.triplets = self.getTriplets()
        self.device = device

        # Raw feature extraction module
        raw_feature_extractor_class = globals()[ architecture['raw_feature_extractor'] ]
        self.raw_feature_extractor = raw_feature_extractor_class()

        # Ranker module
        ranker_ops = []
        if len(self.architecture['ranker'])>0:
            Nfeatures_in = self.raw_feature_extractor.feature_size
            for l in range(len(self.architecture['ranker'])):
                Nfeatures_out = self.architecture['ranker'][l]
                ranker_ops.extend( [ nn.Linear( Nfeatures_in , Nfeatures_out),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(p=self.architecture["dropout"]) ] )
                Nfeatures_in = Nfeatures_out
            ranker_ops.append( nn.Linear(Nfeatures_in, self.architecture['embedding_size']) ) # No activation before embedding        
            self.ranker = nn.Sequential( *ranker_ops )
        else:
            self.ranker = None
            self.architecture['embedding_size']=self.raw_feature_extractor.feature_size
            
    def forward(self, x):
        
        t_stack_list = [] # embeedings for all time points 
        for time_point in range(0,self.architecture['Nt']):
            x_t = x[:,time_point,:,:,:] 
            x_t = self.raw_feature_extractor(x_t)   
            if self.ranker:
                x_t = self.ranker(x_t)
            t_stack_list.append(x_t)
        
        # Keep the time dimension, needed for retreiving triplets later
        t_stack = torch.stack( t_stack_list, dim=1 ) 
        
        # Another view of t_stack where embeddings from all time points are stacked into one vector
        # which corresponds to the embedding of the time course as a whole
        time_course_embedding = t_stack.view(-1,self.architecture['Nt']*self.architecture['embedding_size'])

        return t_stack , time_course_embedding

        
    def getTriplets(self, Ndelta = 2):
        """ Predefines the indexes (along the 'time point' axis of the image batch) of the triplets to be used in the calculation of the ranking loss.
         Triplets consists of (ref, pos, [neg] )
        parameters:
        - Ndelta: Number of time points away from 'ref' to select negatives. (minimum possible value=2)
        If Ndelta=2 , only the first negatives to the rigth and left are selected as posible negatives, i.e., (ref=t2, pos=t3, neg=t4), (ref=t2, pos=t1, neg=t0). Thus triplets come from consecutive time points(harder task)
        If Ndelta=3 , farther away negatives are also selected, i.e., (ref=t2, pos=t3, neg=t4), (ref=t2, pos=t3, neg=t5). 
        """
        triplets = []
        for t in range(self.architecture['Nt']):
            ref = t
            positives_forward  = [ t+i for i in [1]  if ((t+i)<self.architecture['Nt']) ]
            positives_backward = [ t+i for i in [-1] if ((t+i)>=0)  ]
            negatives_forward  = [ t+i for i in list(range(2, Ndelta+1)) if ((t+i)<self.architecture['Nt']) ]
            negatives_backward = [ t+i for i in list(range(-Ndelta,-1))  if ((t+i)>=0)  ]
            for pos in positives_forward:
                for neg in negatives_forward:
                    triplets.append( (ref,pos,neg))
            for pos in positives_backward:
                for neg in negatives_backward:
                    triplets.append( (ref,pos,neg))
        return triplets
            
    def trainableParametersSummary(self):
        """
        Print all tranable parameters
        """
        param_sum =0
        out_str = 'List of trainable parameters:\n'
        for name, param in self.named_parameters():
            if param.requires_grad:
                out_str = out_str + name + ':' + str(param.data.shape) +'\n'
                param_sum = param_sum + param.numel()
        param_sum = param_sum*1E-6
        out_str = out_str + 'Total parameters: '+str(param_sum)+' million \n'
        logging.info(out_str)
        
    def to(self, *args, **kwargs):
        """ Wraper to add/change a device atribute to the model class everytime
        the model is sent to a different device"""
        super(TimeCourseTripletNet, self).to(*args, **kwargs)
        self.device = list(self.parameters())[0].device
    
    def findPairDistance(self, features_1, features_2 , distance_type):
        # Fisrt check whether it is a single example or a batch of example.
        # Necesary to decide along which dimensions the norm is taken 
        is_batched = features_1.ndimension() == 2 
        if is_batched:
            reduce_dim = 1
        else:
            reduce_dim = 0
        if distance_type in ['euc','euclidian']:
            D = torch.norm( features_1 - features_2, p=2 , dim=reduce_dim )**2 
        elif distance_type in ['cos','cosine']:
            D = 1-F.cosine_similarity(features_1,features_2, dim=reduce_dim)
        else:
            raise ValueError('Error in findPairDistance(). Unkown distance type')
        return D
                        
    def getTimeCourseRankingLoss(self, t_stack):
        loss = 0
        if ( ('distance_type' in self.train_params.keys()) & ('ranking_gap' in self.train_params.keys()) )==False:
            raise ValueError('Error in getTimeCourseRankingLoss(). distance_type and/or ranking_gap are unset. Please pass a valid train_params dictionary')
        for ref, pos, neg in self.triplets:
            D_pos = self.findPairDistance( t_stack[:,ref,:], t_stack[:,pos,:], self.train_params['distance_type'])
            D_neg = self.findPairDistance( t_stack[:,ref,:], t_stack[:,neg,:], self.train_params['distance_type'])
            D = D_pos - D_neg 
            rank_error  = torch.max( D + self.train_params['ranking_gap'] , torch.zeros_like(D) ) # size = batch_size
            loss = loss + torch.sum( rank_error ) # sum over al batch samples
        # Divide loss by the number of samples conisdered
        batch_size = t_stack.shape[0]
        loss = loss/( len(self.triplets)*batch_size )
        return loss        

class RawFeatureExtractorAlexNet(nn.Module):
    
    """ 
    Transfer Learning model based on Alexnet
    Uses the first fully connected layer of a pretrained Alexnet to extract 4096 features from an input image
    """
    
    def __init__(self):
        super(RawFeatureExtractorAlexNet, self).__init__()
        
        # Feature extraction of individual time points 
        alexnet = models.alexnet(pretrained=True)
        self.conv_features = nn.Sequential( *list(alexnet.features.children()) )
        self.partial_classifier   = nn.Sequential( *list(alexnet.classifier.children())[1:3] )
        for param in self.conv_features.parameters(): 
            param.requires_grad = False
        for param in self.partial_classifier.parameters(): 
            param.requires_grad = False
        self.trainable =False
        self.feature_size = list(self.partial_classifier.children())[0].out_features
    
    def unfreezeParams(self):
        for param in self.conv_features.parameters(): 
            param.requires_grad = True
        for param in self.partial_classifier.parameters(): 
            param.requires_grad = True
        self.trainable =True
    
    def freezeParams(self):
        for param in self.conv_features.parameters(): 
            param.requires_grad = False
        for param in self.partial_classifier.parameters(): 
            param.requires_grad = False
        self.trainable = False

    def forward(self, x):
        features = self.conv_features(x)
        features = features.view(features.size(0), 256 * 6 * 6)
        features = self.partial_classifier(features)    
        return features

        
class RawFeatureExtractorInceptionv3(nn.Module):
    
    """ 
    Uses the last layer before the fully connected layer of inception architecture to extract features from an input image
    """
    
    def __init__(self):
        super(RawFeatureExtractorInceptionv3, self).__init__()
        
        # Feature extraction of individual time points 
        self.inception = models.inception_v3(pretrained=True)
        
        # Set off gradient computations and set inception model to eval (Important to set all dropout to zero, 
        # dont update batch norm runing stats, and not to output auxiliary classifiers)
        self.inception.eval()  
        for param in self.inception.parameters():
            param.requires_grad = False
        self.feature_size =self.inception.fc.in_features
    
    def unfreezeParams(self):
        for param in self.inception.parameters(): 
            param.requires_grad = True
        self.trainable =True
    
    def freezeParams(self):
        for param in self.inception.parameters(): 
            param.requires_grad = False
        self.trainable = False

    def forward(self, x):
        """
        copied from https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
        """
        # 299 x 299 x 3
        x = self.inception.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.inception.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.inception.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.inception.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.inception.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.inception.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.inception.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.inception.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.inception.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.inception.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.inception.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.inception.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.inception.Mixed_6e(x)
        # 17 x 17 x 768
        #if self.inception.training and self.inception.aux_logits:
            #aux = self.inception.AuxLogits(x)
        # 17 x 17 x 768
        x = self.inception.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.inception.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.inception.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        return x

class TVNScaler: 

    """
    Scales features based on a typical variance normalization transform calcualted from a control dataset
    """
    def __init__(self, epsilon_variance_ratio = 0.01):
        self.epsilon_variance_ratio = epsilon_variance_ratio
        
    def fit(self,ctrl_features):
        
        # First check if you have enough samples before doing the PCA
        enough_ctrl_samples = ctrl_features.shape[0] >= ctrl_features.shape[1]
        assert enough_ctrl_samples, ('Number of control samples should be >= than the time-course feature size (at least %i control time-courses needed)'%ctrl_features.shape[1])
        
        self.epsilon = self.epsilon_variance_ratio*np.std(ctrl_features)
        self.pca_model = PCA(n_components = ctrl_features.shape[1])  
        ctrl_features_pca = self.pca_model.fit_transform(ctrl_features)
        self.means = np.mean( ctrl_features_pca , axis = 0)
        self.stds  = np.std(  ctrl_features_pca , axis = 0)
    
    def transform(self, features):
        transformed_features =  (self.pca_model.transform(features) - self.means)/(self.stds + self.epsilon)           
        return transformed_features
    
    
class TVNClusterer: 
    """
    Overarching object which puts together a scaler and a clustering model.
    Besides predicting hard clusters, this class uses the objective function of fuzzy kmeans to predict fuzzy clusters (even though the clustering model was not trained on a fuzzy objective). 
    The strength of the fuzzyfication is determined by fuzzyness_param ( should be > 1). The closer it is to 1, the closter to hard kmeans. The bigger fuzzyness_param, the fuzzier (converge to the global cluster).
    """
    
    def __init__(self, scaler , clusterer , cluster_annotations=[] ):
        self.scaler = scaler   
        self.clusterer = clusterer 
        self.fuzzyness_param = 1.1 # >1 bigger = fuzzier
        self.cluster_annotations = cluster_annotations

    def predictClusters(self,raw_features):
        self.checkSizeCompatibility( raw_features.shape[1] )
        norm_features = self.scaler.transform( raw_features )
        clusters = self.clusterer.predict( norm_features )
        return clusters
        
    def predictFuzzyClusters(self, raw_features):
        """
        Maybe change by a softmax function ? 
        Code taken from the implementation of sklearn-extensions 0.0.2 
        https://gist.github.com/mblondel/1451300
        see method _e_step() from the FuzzyKMeans class.
        """
        self.checkSizeCompatibility( raw_features.shape[1] )
        norm_features = self.scaler.transform( raw_features )
        D = 1.0 / self.clusterer.transform( norm_features )**2 # transform to distance space
        D **= 1.0 / (self.fuzzyness_param - 1)
        fuzzy_clusters = D/np.sum(D, axis=1)[:, np.newaxis]
        return fuzzy_clusters
    
    def sortCentroids( self, ref_centroid ):
        """
        Exchanges centroids such that they are sorted by increasing distance to ref centroid"
        """
        centroids = self.clusterer.cluster_centers_    
        centroid_distances = spatial.distance.cdist( centroids , centroids )
        D_ref = centroid_distances[ref_centroid,:] # distance relative to the ref cluster
        sorted_idx = np.argsort(D_ref)
        temp = centroids.copy()
        for i, idx in enumerate(sorted_idx):
            centroids[i,:]= temp[idx,:]
        self.clusterer.cluster_centers_ = centroids 
    
    def checkSizeCompatibility(self, features_size):
        """"
        Since embeddings migth be translated (or not) to the first time point, this check reminds the user to take care of translating (or not) the features
        """
        size_check = features_size == self.clusterer.cluster_centers_.shape[1]
        assert size_check, ('Feature size (%i) is not compatible with the clustering model feature size (%i). This migth be caused by: \n- Inclusion/exclusion of embeddings substraction from t0 \n- Clustering model migth have been trained on time-course embeddings from different time points'%( features_size, self.clusterer.cluster_centers_.shape[1]) )
          
