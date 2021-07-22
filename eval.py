import torch
from torchvision import datasets, models, transforms
import torch.utils.data as data
import multiprocessing
from sklearn.metrics import confusion_matrix
import sys
import matplotlib.pyplot as plt
import seaborn as sn
import pdb
import numpy as np
import tqdm
import cv2
from sklearn.manifold import TSNE

"""
python eval.py eval_dir expt_name model_checkpoint/model.pth

"""

# Paths for image directory and model
EVAL_DIR=sys.argv[1]
exp_name=sys.argv[2]
EVAL_MODEL=sys.argv[3]
PLOT_TITLE="Plots_" + sys.argv[2]
# Load the model for evaluation
model = torch.load(EVAL_MODEL)
model.eval()

# Configure batch size and nuber of cpu's
num_cpu = multiprocessing.cpu_count()
bs = 8
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Prepare the eval data loader
eval_transform=transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

eval_dataset=datasets.ImageFolder(root=EVAL_DIR, transform=eval_transform)
eval_loader=data.DataLoader(eval_dataset, batch_size=bs, shuffle=False,
                            num_workers=num_cpu, pin_memory=True)

# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Number of classes and dataset-size
num_classes=len(eval_dataset.classes)
dsize=len(eval_dataset)

# Class label names
class_names=['Dense Traffic', 'Sparse Traffic']

# Initialize the prediction and label lists
predlist=torch.zeros(0,dtype=torch.long, device='cpu')
lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

images_list = []
features = []
gts = []

# Evaluate the model accuracy on the dataset
correct = 0
total = 0

with torch.no_grad():
    """
    The main evaluations scipt to calculate number of correct predictions
    """
    for images, labels in eval_loader:
        images, labels = images.to(device), labels.to(device)
        n, nc, ht, wd = images.shape
        feats, outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
        features.extend(feats.data.cpu().numpy())
        gts.extend(labels.data.cpu().numpy())
        
        ## Preprocessing to visualiize images for TSNE plots
        images_raw = images.data.cpu().numpy()
        mean = np.tile(np.array([0.485, 0.456, 0.406]), (1, 1, 1, 1)).transpose(0,3,1,2)
        std = np.tile(np.array([0.229, 0.224, 0.225]), (1, 1, 1, 1)).transpose(0,3,1,2)
        # print(std.shape, mean.shape, images_raw.shape)

        images_raw = np.clip((images_raw * std) + mean, 0,1)*255 
        images_list.extend(images_raw)

        predlist=torch.cat([predlist,predicted.view(-1).cpu()])
        lbllist=torch.cat([lbllist,labels.view(-1).cpu()])


### Logging Reuslts

# Overall accuracy
overall_accuracy=100 * correct / total
print('Accuracy of the network on the {:d} test images: {:.2f}%'.format(dsize, 
    overall_accuracy))


# Save Confusion matrix
conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
print('Confusion Matrix')
print('-'*16)
print(conf_mat,'\n')

plt.figure(figsize = (5,5))
plt.title(PLOT_TITLE, fontsize =20)
heatmap = sn.heatmap(conf_mat, annot=True, cmap='viridis', linecolor='white', linewidths=1, xticklabels=class_names, yticklabels=class_names, fmt='d')

figure = heatmap.get_figure()
## NOTE: Change the filename
figure.savefig(str(exp_name)+'ConfusionMatrix.png', dpi=400)

# Per-class accuracy
class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
print('Per class accuracy')
print('-'*18)

for label, accuracy in zip(eval_dataset.classes, class_accuracy):
     print('Accuracy of class %8s : %0.2f %%'%(label, accuracy))

## Plot TSNE Features:
#label 0 = dense and 1 = sparse

colors_per_class = {'sparse_traffic' : [70, 227, 175], 'dense_traffic' : [16, 58, 254]}

tsne = TSNE(n_components=2).fit_transform(np.array(features))

def scale_distrib(x):
    scale_range = (np.max(x)-np.min(x))
    start = x - np.min(x)
    return start/scale_range

def plot_tsne_features(x, y, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for label in colors_per_class:
        if label == 'sparse_traffic':
            label_id = 1
        elif label == 'dense_traffic':
            label_id = 0
        else:
            print("Invalid Label!!")

        indices = [i for i, l in enumerate(labels) if l == label_id]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255
        ax.scatter(current_tx, current_ty, c=color, label=label)

    ax.legend(loc='best')
    plt.show()
    plt.savefig(str(exp_name)+"_TSNE_Features.png")


def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.transpose(1,2,0).shape

    center_x = int(image_centers_area_size * x) + offset
    center_y = int(image_centers_area_size * (1 - y)) + offset

    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height
    return tl_x, tl_y, br_x, br_y


def plot_tsne_images(tx, ty, images, labels, plot_size=1000, max_image_size=100):
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset
    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)
    colors_per_class =  [[16, 58, 254], [70, 227, 175]]
    _, ht, wd = images[0].shape

    for im, lbl, x, y in zip(images, labels, tx, ty):
        im = im.transpose(1,2,0).copy()
        im = cv2.rectangle(im, (0,0), (wd-1, ht-1), color=colors_per_class[lbl], thickness=10)
        im = im.transpose(2,0,1).copy()
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(im, x, y, image_centers_area_size, offset)
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = im.transpose(1,2,0)

    cv2.imwrite(str(exp_name)+"TSNE_images.png", tsne_plot)


tx = scale_distrib(tsne[:,0])
ty = scale_distrib(tsne[:,1])

plot_tsne_features(tx, ty, gts)
plot_tsne_images(tx, ty, images_list, gts, 2048, 224)

print("Evaluation completed")
