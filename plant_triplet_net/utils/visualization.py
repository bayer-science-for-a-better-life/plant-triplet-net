import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from PIL import Image, ImageEnhance

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
random_seed = 37

def plotTrainingHistory(loss_history, title='', show=False):
    """
    Displays the different losses used in the optimization of a neural network parameters
    """
    fig = plt.figure(num=None, figsize=(14, 5), facecolor='w', edgecolor='k')
    y_lims = []
    plt.subplot(2,2,1)
    plt.plot(loss_history['minibatch']['train'],'k.-',markersize=1)
    legend = ['train minibatch']
    plt.legend(legend,fontsize=13)
    plt.xlabel('step',fontsize=13)
    plt.ylabel('loss',fontsize=13)
    plt.grid()
    if len(y_lims)>0: plt.ylim(y_lims)
    plt.subplot(2,2,2)
    plt.plot(loss_history['minibatch']['val'],'k.-',markersize=1)
    legend = ['val minibatch']
    plt.legend(legend,fontsize=13)
    plt.xlabel('step',fontsize=13)
    plt.ylabel('loss',fontsize=13)
    if len(y_lims)>0: plt.ylim(y_lims)
    plt.grid()
    plt.subplot(2,2,3)
    plt.plot(loss_history['epoch']['train'],'k.-')
    plt.legend(['train epoch'],fontsize=13)
    plt.xlabel('epoch',fontsize=13)
    plt.ylabel('loss',fontsize=13)
    if len(y_lims)>0: plt.ylim(y_lims)
    plt.grid()
    plt.subplot(2,2,4)
    plt.plot(loss_history['epoch']['val'],'k.-')
    plt.legend(['val epoch'],fontsize=13)
    plt.xlabel('epoch',fontsize=13)
    plt.ylabel('loss',fontsize=13)
    if len(y_lims)>0: plt.ylim(y_lims)
    plt.grid()
    
    fig.suptitle(title,fontsize=15) 
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if show:
        plt.show()
    return fig

def plotTimeCoursesGrid(file_paths, Ncols=5, frame_color=[0,0,0], show=True, fig_size = [], title = []):
    
    if len(file_paths)%Ncols == 0:
        Nrows =  int(len(file_paths)/Ncols)
    else: 
        print('Warning. Number of columns is not dividible by the number of files')
    
    if len(fig_size)==0: fig_size = (Ncols*1.0,Nrows*1.0 )
    fig = plt.figure(num=None, figsize=fig_size, facecolor='w', edgecolor='k')
    
    # Collect all images into a grid
    k = 0
    for r in range(Nrows):
        for c in range (Ncols):
            im =  Image.open(file_paths[k]) 
            if True:
                im = ImageEnhance.Brightness(im).enhance(4)
            im = np.asarray(im)
            if c == 0: grid_row = im
            else: grid_row = np.concatenate([grid_row,im], axis = 1 )
            k = k+1
        if r == 0: im_grid = grid_row
        else: im_grid = np.concatenate([im_grid,grid_row], axis = 0 )
    
    ax = plt.subplot(1,1,1)
    plt.imshow(im_grid)
    plt.xticks([])
    plt.yticks([])
    for s in ax.spines.values(): s.set_color(frame_color)
    for s in ax.spines.values(): s.set_linewidth(4)  
    k = k+1
    if len(title)>1:
        plt.title(title)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plotEmbeddings(embeddings, show=False):
    fig = plt.figure(figsize=(15,3))
    fig.patch.set_facecolor('w')
    plt.subplot(1,2,1)
    Nsamples = min(100, embeddings.shape[0])
    for i in range(Nsamples): plt.plot(embeddings[i,:],'.')
    plt.xlabel('Feature idx')
    plt.ylabel('Feature value')
    plt.subplot(1,2,2)
    y = embeddings[0:Nsamples,:].reshape(-1)
    n, bins, patches = plt.hist(y, 50, normed=1, facecolor='green', alpha=0.75)
    plt.xlabel('Feature value')
    plt.ylabel('Frequency')
    fig.tight_layout()
    if show:
        plt.show()
    return fig
    
def getSpacedColors(Ncolors):
    cmap = matplotlib.cm.get_cmap('winter')
    G_trans = 0.37  # translation needed to put cluster 0 in green for gist_ncar colormap
    delta = (1-G_trans)/Ncolors
    spaced_colors = np.array(range(Ncolors))*delta + G_trans
    spaced_colors = cmap(spaced_colors)
    spaced_colors = spaced_colors[:,0:3] # remove the alpha component
    return spaced_colors

def drawMST(centroid_distances, cluster_colors, figsize = (8,8), draw_numbers =True, node_size=400, show=False, title =''):
    
    Nclusters = centroid_distances.shape[0]
    
    # Create a fully conected  graph object form centroid distances 
    G   = nx.Graph(centroid_distances)
    mst = nx.minimum_spanning_tree(G) 
    
    # re-format labels as a dictionary (needed by nx.draw_networkx_labels)
    labels = dict()
    for c in range(Nclusters): labels[c] = str(c)

    # Plot tree 
    fig, ax = plt.subplots(num=None, figsize=figsize, facecolor='w', edgecolor='k')
    nodes_pos = nx.kamada_kawai_layout(mst)  # positions for all nodes
    nx.draw_networkx_nodes      ( mst, nodes_pos, nodelist=mst.nodes, node_color=cluster_colors, alpha=1, node_size=node_size)
    nx.draw_networkx_edges      ( mst, nodes_pos, edgelist=mst.edges, edge_color='k'           , alpha=1)
    
    if draw_numbers:
        text = nx.draw_networkx_labels     ( mst, nodes_pos, labels=labels, font_size=10)
        for _,t in text.items():
            t.set_rotation(0)
    plt.title(title,fontsize=15)
    plt.xticks([])
    plt.yticks([])
    
    if show:
        plt.show()
        
    return fig

def plotClusterImages(results, sample_selection,cluster_colors, show=False):
    
    fig_list = []
    title_list = []
    for cluster in sample_selection['selected_clusters']:
        
        fuzzy_cluster_col = 'fuzzy_cluster_'+('0'+str(cluster))[-2:]
        select_idx = (results['cluster']==cluster) & (results[fuzzy_cluster_col]>sample_selection['min_confidence'])
        if (select_idx.sum()>0):
            # Select subset of images to plot
            if sample_selection['top_samples']: 
                select_idx = (results['cluster']==cluster)
                temp = results.loc[select_idx].sort_values(by=fuzzy_cluster_col, ascending=False)
                select_idx = temp.index.values[0:min(sample_selection['N_per_cluster'],len(select_idx))]
                samples = results.loc[select_idx]
                title = 'Top samples cluster '+str(cluster)+' (min_confidence '+str(sample_selection['min_confidence'])+')'
            else:
                samples = results.loc[select_idx].sample(min(sample_selection['N_per_cluster'],select_idx.sum()), random_state=random_seed)
                title = 'Random samples cluster '+str(cluster)+' (min_confidence '+str(sample_selection['min_confidence'])+')'       
            # Get list of filenames 
            filenames = []            
            for t0_file in samples['file_path'].values:
                filenames.extend( [t0_file.replace('_t0','_t'+str(t)) for t in sample_selection['time_points']])            
            # Plot images 
            fig = plotTimeCoursesGrid( filenames, Ncols=len(sample_selection['time_points']), 
                                         frame_color=cluster_colors[cluster], show=False, title=title )
            fig_list.append(fig)
            title_list.append(title)
            if show:
                plt.show()
            
    return fig_list, title_list

def plotClusterDistributions(results, cluster_colors, title='', show=False):
    fig, _ = plt.subplots(num=None, figsize=(10,4), facecolor='w', edgecolor='k')
    Nclusters = len(cluster_colors)
    cluster_counts = results.groupby(by='cluster').count().values[:,0]   
    clusters = results.groupby(by='cluster').count().index.values
    for c, counts in zip(clusters,cluster_counts):
        plt.bar(c,counts,width=1, color = cluster_colors[c])
    plt.xlabel('Cluster',fontsize=15)
    plt.ylabel('Num samples',fontsize=15)
    plt.xticks(np.array(range(Nclusters)), range(Nclusters))
    plt.title(title,fontsize=15)
    plt.tight_layout()
    if show:
        plt.show()
    return fig