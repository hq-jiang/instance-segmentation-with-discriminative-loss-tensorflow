import os
import json
import csv
import glob
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np

def read_json(data_dir, json_string):
    json_paths = glob.glob(os.path.join(data_dir,json_string))
    print json_paths
    data = []
    for path in json_paths:
        with open(path) as f:
            d = (line.strip() for line in f) #Erzeugt Generator Objekt
            d_str = "[{0}]".format(','.join(d))
            data.append(json.loads(d_str))

    num_samples = 0
    for d in data:
        num_samples += len(d)
    print 'Number of labeled images:', num_samples
    print 'data keys:', data[0][0].keys()
    
    return data

def read_image_strings(data, input_dir):
    img_paths = []
    for datum in data:
        for d in datum:
            path = os.path.join(input_dir, d['raw_file'])
            img_paths.append(path)
    
    num_samples = 0
    for d in data:
        num_samples += len(d)
    assert len(img_paths)==num_samples, 'Number of samples do not match'
    print img_paths[0:2]
    
    return img_paths

def save_input_images(output_dir, img_paths, mode):
    for i, path in tqdm(enumerate(img_paths), total=len(img_paths)):
        img = cv2.imread(path)
        if mode=='train':
            output_path = os.path.join(output_dir,'images', '{}.png'.format(str(i).zfill(4)))
        elif mode=='test':
            output_path = os.path.join(output_dir,'test_images', '{}.png'.format(str(i).zfill(4)))
        #print output_path
        cv2.imwrite(output_path, img)

def draw_lines(img, lanes, height, instancewise=False):
    for i, lane in enumerate(lanes):
        pts = [[x,y] for x, y in zip(lane, height) if (x!=-2 and y!=-2)]
        pts = np.array([pts])
        if not instancewise:
            cv2.polylines(img, pts, False,255, thickness=7)
        else:
            cv2.polylines(img, pts, False,50*i+20, thickness=7)

def draw_single_line(img, lane, height):
    pts = [[x,y] for x, y in zip(lane, height) if (x!=-2 and y!=-2)]
    pts = np.array([pts])
    cv2.polylines(img, pts, False,255, thickness=15)

def save_label_images(output_dir, data, instancewise=True):
    counter = 0

    for i in range(len(data)):
        for j in tqdm(range(len(data[i]))):
            img = np.zeros([720, 1280], dtype=np.uint8)
            lanes = data[i][j]['lanes']
            height = data[i][j]['h_samples']
            draw_lines(img, lanes, height, instancewise)
            output_path = os.path.join(output_dir,'labels', '{}.png'.format(str(counter).zfill(4)))
            cv2.imwrite(output_path, img)
            counter += 1

mode='train'
input_dir = './train_set'
json_string = 'label_data_*.json'
output_dir = '.'

if not os.path.isdir('images'):
	os.mkdir('images', 0775)
if not os.path.isdir('labels'):
	os.mkdir('labels', 0775)

data = read_json(input_dir, json_string)
img_paths = read_image_strings(data, input_dir)

#save_input_images(output_dir, img_paths, mode)
save_label_images(output_dir, data)