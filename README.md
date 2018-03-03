## Instance Segmentation with a Discriminative Loss Function

Tensorflow implementation of [Semantic Instance Segmentation with a Discriminative Loss Function](https://arxiv.org/abs/1708.02551) trained on the [TuSimple dataset](http://benchmark.tusimple.ai/#/t/1)

---
### Files
├── __data__ here the data should be stored  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   └── __tusimple_dataset_processing.py__ processes the TuSimple dataset  
├── __inference_test__ inference related data  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   └── __images__ for testing the inference    
├── __trained_model__  pretrained model for finetuning  
├── __clustering.py__ mean-shift clustering  
├── __datagenerator.py__ feeds data for training and evaluation  
├── __enet.py__ [Enet architecture](https://github.com/kwotsin/TensorFlow-ENet)  
├── __inference.py__ tests inference on images  
├── __loss.py__ defines discriminative loss function  
├── __README.md__  
├── __training.py__ contains training pipeline  
├── __utils.py__ contains utilities files for building and initializing the graph  
└── __visualization.py__ contains visualization of the clustering and pixel embeddings  


### Instructions

#### Inference
1. To test the inference of the trained model execute:  
`python inference.py --modeldir trained_model`

#### Training

1. Download the [TuSimple training dataset](http://benchmark.tusimple.ai/#/t/1) and extract its contents to the `data` folder.
2. Run the following script to prepare images and labels.  
`python data/tusimple_dataset_processing.py <train_data_dir>`  
This should create the following folder structure:  
├── data  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── images  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── labels  
3. For training on the dataset execute:  
`python training.py`  
alternatively use optional parameters (default parameters in this example):   
`python training --srcdir data --modeldir pretrained_semantic_model --outdir saved_model --logdir log --epochs 50 --var 1.0 --dist 1.0 --reg 1.0 --dvar 0.5 --ddist 1.5
`
4. To test the trained network execute:
`python inference.py --modeldir saved_model`

#### Todo
- pip requirements
- writeup
- images

Tensorflow version 1.2

### References
Credits to this [Enet](https://github.com/kwotsin/TensorFlow-ENet) implementation  
[TuSimple dataset](http://benchmark.tusimple.ai/#/t/1)
