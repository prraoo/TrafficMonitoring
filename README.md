# TrafficMonitoring

This is an implementation of the Traffic Monitoring Project as a part of fulfillment of the data science course. 

## File Organisation

The various Python scripts assume a shared organizational structure such that the output from one script can easily be used as input to another. 

### Dataset Structure
The TrafficNet-V2 dataset is organised as follows:
```
data
├── test
│   ├── dense_traffic
│   └── sparse_traffic
├── train
│   ├── dense_traffic
│   └── sparse_traffic
└── val
    ├── dense_traffic
    └── sparse_traffic
```
This is format is directly readable by Pytorch dataloaders and we this the same structure in our project.

## Training a model
Once the data is organized, models can be trained as follows:
```
python train.py \
 --mode=scratch \
 --batch_size=64 \
 --num_workers=8 \
 --num_classes=2 \
 --save_path=saved_models/trafficnet_V2.pth \
 --comment=demo \
 --scheduler_step=10 \
```
## Training WGAN
In order to train the generator for WGAN and save it, run
```
python wgan.py --n_epochs=20000 --img_size=224 --channels=3
```

The model with minimum validation loss will be saved in the `saved_models` folder. In addition the training and validation progress are saved as tensorboard logs in the `runs` folder.  

## Visualize tensorboard logging:
To visualize the training progress, install tensorboard package and execute the command
```
tensorboard --logdir runs/
```

## Evaluation
The final model evaluations can be done on the testset as following
```
python eval.py data/test/ 'Demo-Results' saved_models/trafficnet_V2.pth
```
## Team

Pramod Rao, Sven Krichner, Yashaswini Kumar

## Acknowledgements

We want to acknowledge the help of following repositories where we take their implementations and build our codes and dataset
1. Dataset: https://github.com/OlafenwaMoses/Traffic-Net
2. VGG16: https://github.com/msyim/VGG16
3. Center Loss: https://github.com/KaiyangZhou/pytorch-center-loss
4. WGAN: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/wgan



