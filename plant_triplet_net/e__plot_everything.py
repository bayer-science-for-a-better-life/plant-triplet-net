from plant_triplet_net.utils import utils
from plant_triplet_net.utils import visualization as dv
import os
from scipy import spatial
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import json
import time
import logging

def reportFigure(fig,title,pdf,output_dir):
    # save to the pdf summary file
    pdf.savefig(fig) 
    # Save figure as an image
    fig_file = os.path.join( output_dir, title+'.png') 
    fig.savefig(fig_file)
    # Close to avoid accumulation in memmory
    plt.close(fig)   

def main(feature_extraction_model_file, features_file, clustering_model_file, cluster_results_file, cluster_plot_params, output_dir, log_output_file):
    
    if log_output_file: 
        utils.ensureDir(log_output_file)
        logging.basicConfig(filename=log_output_file, level=logging.INFO, format ='%(asctime)s  - %(levelname)s - %(message)s')
    else: logging.basicConfig(level=logging.INFO)
    logging.info('\n\n-------------- Started plotting --------------\n')
    print('Started plotting')
    start_time = time.time()
    
    # Open pdf 
    summary = PdfPages(os.path.join(output_dir,'summary.pdf'))
    
    # Visualizations related to the feature extraction model
    if len(feature_extraction_model_file)>0:
        model_info = utils.loadFeatureExtractionModel(feature_extraction_model_file, only_info = True)
        title =  'training history ' +os.path.basename(feature_extraction_model_file)[0:-4]
        fig = dv.plotTrainingHistory(model_info['loss_hist'],title=title,show=False)
        reportFigure(fig,title,summary,output_dir)

    # Visualization fo extracted fatures
    features, metadata, time_point_feature_size = utils.loadEmbeddings(features_file)
    fig = dv.plotEmbeddings(features)
    title =  'Embeddings ' +os.path.basename(features_file)[0:-4]
    reportFigure(fig,title,summary,output_dir)

    # Visualizations related to the clustering model  
    if len(clustering_model_file)>0:
        model = utils.loadClusteringModel(clustering_model_file)
        centroids = model.clusterer.cluster_centers_
        centroid_distances = spatial.distance.cdist( centroids , centroids )
        Nclusters = centroids.shape[0]
        cluster_colors = dv.getSpacedColors(Nclusters)
        title = 'clusters minimum spanning tree' 
        fig = dv.drawMST(centroid_distances, cluster_colors, title=title , show=False)
        reportFigure(fig,title,summary,output_dir)

        # Visualizations related to the clustering Results  
        # Due tos lazyness the scripts below need to have acces to the total number of clusters in the model (Nclusters)
        # ortherwise plotting will fail if we dont have at least one sample for all the clusters
        # !!!!! Fix me later
        if len(cluster_results_file)>0:

            results = pd.read_csv(cluster_results_file,index_col=0)
            if not cluster_plot_params['selected_clusters']:
                cluster_plot_params['selected_clusters'] = range(Nclusters)

            # Top OR Random images per cluster
            fig_list, title_list = dv.plotClusterImages(results, cluster_plot_params,cluster_colors,show=False)
            for i, title in enumerate(title_list) :
                fig = fig_list[i] # fig needs to be explicitely called here. Don't iterate over fig_list directly 
                reportFigure(fig,title,summary,output_dir)

            # Sample histogram distribution 
            title = 'cluster distribution'
            fig = dv.plotClusterDistributions(results, cluster_colors, title=title, show=False)
            reportFigure(fig,title,summary,output_dir)

    # Close pdf
    summary.close()

    t_elapsed = (time.time() - start_time)/60
    logging.info("\n\ntime elapsed: %.1f minutes "% t_elapsed)
    logging.info("\n-------------- Completed plotting --------------\n")
    print('Ended features to plotting')
    
    
if __name__ == '__main__':

    # Get name of the file with all the input settings 
    parser = argparse.ArgumentParser()
    parser.add_argument('-i' ,  help='File with all the inputs' )
    args = parser.parse_args()
    
    # Read json file and
    inputs = json.load(open(args.i))    
    
    # General params
    feature_extraction_model_file = inputs['feature_extraction_model_file']
    features_file = inputs['features_file']
    clustering_model_file = inputs['clustering_model_file']
    cluster_results_file = inputs['cluster_results_file']
    cluster_plot_params = inputs['cluster_plot_params']
    required_keys = ['selected_clusters','time_points','min_confidence','N_per_cluster','top_samples']
    utils.checkKeys(cluster_plot_params, required_keys)

    output_dir = inputs['output_dir']
    utils.ensureDir(output_dir)
    if len(os.listdir(output_dir))>0: 
        raise OSError('Output directory is not empty')
    log_output_file = inputs['log_output_file']
    
    main(feature_extraction_model_file, features_file, clustering_model_file, cluster_results_file, cluster_plot_params, output_dir, log_output_file)



