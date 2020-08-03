from PIL import Image
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from random import randint
import logging

# Create data set class: override methods __len__ (len()) and __getitem__ ( [] )
class TimeCourseImageDataset(Dataset):
    """
    Loader for stacks of time-course images. One time course corresponds to all the images sharing the same 'experiment_ID','plate_ID' and 'well_ID' values.  
    """
    
    def __init__(self, dataset_info, selected_time_points, image_size, transform=None, augmentation=True, time_course_rotation = False ):
        """
        :param dataset_info: pandas dataframe containing the following columns: 'experiment ID','plate ID', 'well ID', 'time point','file name'
        :param selected_time_points: list of time points to be considered  
        :param image_size: int number of pixels. Only squared image supported so far
        :param transform: pythorch transform object to pre-process the images
        :param augmentation: apply random angle rotation, horizontal and vertical flip. All images from the same time course undergo the same transformations
        :param time_course_rotation: Randomly rotate images within the the same time course
        :return: image stack of size = [batch_size, Nt, Nchann, im_size, im_size]
        """
        self.transform = transform
        self.selected_time_points = selected_time_points
        self.Nt = len(self.selected_time_points)
        self.image_size = image_size
        self.Nchann = 3
        self.augmentation = augmentation
        self.time_course_rotation = time_course_rotation  # Randomly rotate images from the the same time course 
        
        # Group images corresponding to the same experimental sample: one group = all time point images from the same sample
        sample_ID_cols = ['experiment_ID','plate_ID','well_ID']
        self.time_course_groups = list( dataset_info.groupby(sample_ID_cols))
        # Fetch metadata by getting the columns of the first time point. 
        self.groups_metadata = dataset_info.groupby(sample_ID_cols).apply(lambda x: x.sort_values(by='time_point').iloc[0]) 
        # Drop the time_point column since it has no meaning anymore. Also drop sample_ID_cols to be able to reset the index
        # After reseting, the index from the metadata is algined with the time_course_groups index
        self.groups_metadata = self.groups_metadata.drop(columns=sample_ID_cols+['time_point']).reset_index() 
           
    def __len__(self):
        return len(self.time_course_groups)
    
    def __getitem__(self, idx):
        
        df = self.time_course_groups[idx][1].sort_values(by='time_point') # data frame with filenames
        image_t_stack = torch.zeros((self.Nt,self.Nchann,self.image_size,self.image_size),dtype=torch.float32)
        
        # get random parameters for augmentation. All images from the same time series
        # should undergo the same augmentaiton transformations
        rand_angle = randint(0,360)
        ver_flip  = bool(randint(0,1))
        hor_flip   = bool(randint(0,1))
        
        # Load and transfrom stack of images from the selected time course.
        # If a time point is not available, corresponding image will be zeros
        for i, time_point in enumerate(self.selected_time_points):
            
            # Get image data
            filename = df.loc[ df['time_point'] == time_point, 'file_path'].values
            if len(filename)==1:
                image = Image.open(filename[0]) # shape = (Chan, H , W )
                if self.augmentation:
                    if self.time_course_rotation:
                        rand_angle = randint(0,360)
                    image = transforms.functional.rotate(image, rand_angle, resample=Image.BILINEAR)
                    if ver_flip: 
                        image = transforms.functional.vflip(image)
                    if hor_flip: 
                        image = transforms.functional.hflip(image)
                if self.transform:
                    image = self.transform(image)    
                if not( image.shape[1]==image.shape[2]):
                    raise ValueError('Only squared images are supported in TimeCourseImageDatasets')
                image_t_stack[i,:,:,:]= image
                
            else: 
                print('Warning!!found %i samples for time point %i'%(len(filename),time_point))

        return image_t_stack

def createDataLoaders(datasets_info, selected_time_points, batch_size, image_size,
                                 norm_params = {'means':[0.485, 0.456, 0.406],'stds':[0.229, 0.224, 0.225] },
                                 num_workers = 8 ):
    """
    Create train and validation Time Course Dataloaders
    """
     
    # Default image transformations for Pytorch pretrained models (at least Alexnet VGG and inceptionv3) 
    data_transforms = transforms.Compose([ transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize( norm_params['means'],norm_params['stds'] )])
    datasets = dict()
    dataloaders = dict()
    
    # Train 'time course' dataset, including data augmentation and image rotation between conscutive tiem points 
    drop_last = False
    for set_type in datasets_info.keys():
        if set_type == 'train':
            augmentation = True
            shuffle = True
            time_course_rotation = True
        elif set_type =='val':
            augmentation = False
            shuffle = True
            time_course_rotation = True
        elif set_type == 'eval':
            augmentation = False
            shuffle = False
            time_course_rotation = False
        else:
            raise KeyError('Error in createDataLoaders(). set_type key is unkown. Choose between train, val or eval.')
        datasets[set_type] = TimeCourseImageDataset(datasets_info[set_type], selected_time_points, image_size, transform=data_transforms, augmentation=augmentation, time_course_rotation=time_course_rotation)
        dataloaders[set_type] = DataLoader( datasets[set_type] , batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
        logging.info('Created %s dataloader with %i time-courses'%(set_type,len(datasets[set_type])))
    return dataloaders, datasets



