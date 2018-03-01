import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from clustering import cluster

def save_image_overlay(valid_image, valid_label):

    assert len(valid_image.shape)==3 and len(valid_label.shape)==2, \
        'input dimensions should be [h,w,c]'

    num_unique = np.unique(valid_label)
    blended = valid_image
    for color_id, unique in enumerate(list(num_unique[1:])):
        instance_ind = np.where(valid_label==unique)
        alpha = np.zeros_like(valid_image)
        alpha[instance_ind] = np.array([color_id*70, color_id*70, 255-color_id*50])
        
        blended = cv2.addWeighted(blended, 1, alpha, 1, 0)
    blended = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    cv2.imwrite('overlayed_image.png', blended)   


def evaluate_scatter_plot(log_dir, valid_pred, valid_label, feature_dim, param_string, step):

    assert len(valid_pred.shape)==4 and len(valid_label.shape)==3, \
        'input dimensions should be [b,h,w,c] and [b,h,w]'

    assert valid_pred.shape[3]==feature_dim, 'feature dimension and prediction do not match'


    fig = plt.figure() #plt.figure(figsize=(10,8))
    if feature_dim==2:

        #for i in range(valid_pred.shape[0]):
        #    plt.subplot(2,2,i+1)
        #    #valid_label = valid_label[0]
        #    #print 'valid_pred', valid_pred.shape
        #    #print 'valid_label', valid_label.shape
        #    num_unique = np.unique(valid_label[i])
        num_unique = np.unique(valid_label[0])
        for unique in list(num_unique):
            instance_ind = np.where(valid_label[0]==unique)
            #print 'instance id', instance_ind
            #print valid_pr[instance_ind].shape
            x = valid_pred[0,:,:,0][instance_ind]
            y = valid_pred[0,:,:,1][instance_ind]
            plt.plot(x, y, 'o')
            #plt.imshow(valid_label[i])

    elif feature_dim==3:
        #for i in range(valid_pred.shape[0]):
        #    ax = fig.add_subplot(2,2,i+1, projection='3d')
        #    #valid_pred = valid_pred[0]
        #    #valid_label = valid_label[0]
        ax = fig.add_subplot(1,1,1, projection='3d')
        num_unique = np.unique(valid_label[0])
        colors = [(0., 0., 1., 0.05), 'g', 'r', 'c', 'm', 'y']
        for color_id, unique in enumerate(list(num_unique)):
            instance_ind = np.where(valid_label[0]==unique)
            #print 'instance id', instance_ind
            #print valid_pr[instance_ind].shape
            x = valid_pred[0,:,:,0][instance_ind]
            y = valid_pred[0,:,:,1][instance_ind]
            z = valid_pred[0,:,:,2][instance_ind]
            
            ax.scatter(x, y, z, c=colors[color_id])
    elif feature_dim > 3:
        plt.close(fig)
        return None

    plt.savefig(os.path.join(log_dir, param_string, 'cluster_{}.png'.format(str(step).zfill(6))), bbox_inches='tight')
    plt.close(fig)
