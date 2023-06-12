PATH_TO_REPO = r'C:\Users\hassa\Music\utae-paps-main'
PATH_TO_DATA = r'C:\Users\hassa\Music\PASTIS'
PATH_TO_UTAE_WEIGHTS = r"C:\Users\hassa\Music\UATE_zenodo"
#PATH_TO_UTAEPaPs_WEIGHTS = r"D:\fyp\UTAE+PAPs_PanopticSeg_weights\UTAE_PAPs"
device = 'cpu' # or "cpu"

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import time
from PIL import Image
from flask import Flask, render_template, request, url_for
import os
import torch
import json
import os
from argparse import Namespace

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib

from matplotlib import patches
import numpy as np
import sys
sys.path.append(PATH_TO_REPO) 

from src.dataset import PASTIS_Dataset
from train_panoptic import recursive_todevice
from src.utils import pad_collate
from src.model_utils import get_model

import cv2

import warnings
warnings.filterwarnings('ignore')


# Colormap (same as in the paper)
cm = matplotlib.cm.get_cmap('tab20')
def_colors = cm.colors
cus_colors = ['k'] + [def_colors[i] for i in range(1,19)]+['w']
cmap = ListedColormap(colors = cus_colors, name='agri',N=20)

label_names = [
"Background",
"Meadow",
"Soft winter wheat",
"Corn",
"Winter barley",
"Winter rapeseed",
"Spring barley",
"Sunflower",
"Grapevine",
"Beet",
 "Winter triticale",
 "Winter durum wheat",
 "Fruits,  vegetables, flowers",
 "Potatoes",
 "Leguminous fodder",
 "Soybeans",
 "Orchard",
 "Mixed cereal",
 "Sorghum",
 "Void label"]


# In[3]:


### Utilities 

def load_model(path, device, fold=1, mode='semantic'):
    """Load pre-trained model"""
    with open(os.path.join(path, 'conf.json')) as file:
        config = json.loads(file.read())
    config = Namespace(**config)
    model = get_model(config, mode = mode).to(device)

    sd = torch.load(
        os.path.join(path, "Fold_{}".format(fold+1), "model.pth.tar"),
        map_location=device
        )
    model.load_state_dict(sd['state_dict'])
    return model

def get_rgb(x,b=0,t_show=6):
    """Gets an observation from a time series and normalises it for visualisation."""
    im = x[b,t_show,[2,1,0]].cpu().numpy()
    mx = im.max(axis=(1,2))
    mi = im.min(axis=(1,2))   
    im = (im - mi[:,None,None])/(mx - mi)[:,None,None]
    im = im.swapaxes(0,2).swapaxes(0,1)
    im = np.clip(im, a_max=1, a_min=0)
    return im



def plot_pano_predictions(pano_predictions, pano_gt, ax, cmap=cmap, batch_element=0, alpha=.5):
    pano_instances = pano_predictions['pano_instance'][batch_element].squeeze().cpu().numpy()
    pano_semantic_preds = pano_predictions['pano_semantic'][batch_element].argmax(dim=0).squeeze().cpu().numpy()
    grount_truth_semantic = y[batch_element,:,:,-1].cpu().numpy()

    for inst_id in np.unique(pano_instances):
        if inst_id==0:
            continue # ignore background
        mask = (pano_instances==inst_id)
        try:
            # Get polygon contour of the instance mask
            c,h= cv2.findContours(mask.astype(int), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get the ground truth semantic label of the segment
            u,cnt  = np.unique(grount_truth_semantic[mask], return_counts=True)
            cl = u if np.isscalar(u) else u[np.argmax(cnt)]
            if cl==19: # Not showing predictions for "Void" segments
                continue

            # Get the predicted semantic label of the segment
            cl = pano_semantic_preds[mask].mean()
            color = cmap.colors[int(cl)]
            for co in c[0::2]:
                poly = patches.Polygon(co[:,0,:], fill=True, alpha=alpha, linewidth=0, color=color)
                ax.add_patch(poly)
                poly = patches.Polygon(co[:,0,:], fill=False, alpha=.8, linewidth=4, color=color)
                ax.add_patch(poly)
        except ValueError as e:
            print( cv2.findContours(mask.astype(int), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE))
            
            
def plot_pano_gt(pano_gt, ax, cmap=cmap, batch_element=0, alpha=.5, plot_void=True):
    ground_truth_instances = y[batch_element,:,:,1].cpu().numpy()
    grount_truth_semantic = y[batch_element,:,:,-1].cpu().numpy()

    for inst_id in np.unique(ground_truth_instances):
        if inst_id==0:
            continue  
        mask = (ground_truth_instances==inst_id)
        try:
            c,h= cv2.findContours(mask.astype(int), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
            u,cnt  = np.unique(grount_truth_semantic[mask], return_counts=True)
            cl = u if np.isscalar(u) else u[np.argmax(cnt)]
            
            if cl==19 and not plot_void: # Not showing predictions for Void objects
                continue
            
            color = cmap.colors[int(cl)]
            for co in c[1::2]:
                poly = patches.Polygon(co[:,0,:], fill=True, alpha=alpha, linewidth=0, color=color)
                axes[1].add_patch(poly)
                poly = patches.Polygon(co[:,0,:], fill=False, alpha=.8, linewidth=4, color=color)
                axes[1].add_patch(poly)
        except ValueError as e:
            print(e)


# In[4]:


# Load dataset and models

fold = 3
batch_size = 1
dt = PASTIS_Dataset(folder=PATH_TO_DATA, norm=True,
                target='instance', folds=[fold])
dl = torch.utils.data.DataLoader(dt, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
#iterator =  dl.__iter__()
#device = torch.device(device)


utae = load_model(PATH_TO_UTAE_WEIGHTS, device=device, fold=fold, mode='semantic').eval()
#utaepaps = load_model(PATH_TO_UTAEPaPs_WEIGHTS, device=device, fold=fold, mode='panoptic').eval()


# In[5]:


# Inference on one batch

#batch = recursive_todevice(iterator.__next__(), device)
""""
(x, dates), y = batch
(
    target_heatmap,
    instance_ids,
    pixel_to_instance_mapping,
    instance_bbox_size,
    object_semantic_annotation,
    pixel_semantic_annotation,
) = y.split((1, 1, 1, 2, 1, 1), dim=-1)

im = get_rgb(x, b=0, t_show=2)

plt.imshow(im)

plt.savefig('./static/OriginalS2.png')
"""
BATCH={}

for i in range(0,5):

    iterator =  dl.__iter__()
    device = torch.device(device)


    batch = recursive_todevice(iterator.__next__(), device)
    name='./Test/OriginalS2'+str(i+1)+'.png'
    BATCH['OriginalS2'+str(i+1)+'.png']=batch
    #BATCH.append(batch)
    
    #(x, dates), y = batch(target_heatmap,instance_ids,pixel_to_instance_mapping,instance_bbox_size,object_semantic_annotation,pixel_semantic_annotation,) = y.split((1, 1, 1, 2, 1, 1), dim=-1)
    (x, dates), y = batch
    (
    target_heatmap,
    instance_ids,
    pixel_to_instance_mapping,
    instance_bbox_size,
    object_semantic_annotation,
    pixel_semantic_annotation,
    ) = y.split((1, 1, 1, 2, 1, 1), dim=-1)
    

    im = get_rgb(x, b=0, t_show=2)

    plt.imshow(im)
    
    plt.savefig(name)

    

# In[ ]:





# In[6]:

batch = 0

app = Flask(__name__)  # create the Flask app


# In[7]:

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")


NAME = str()

@app.route("/deepfarm",methods=['GET', 'POST'])
def deepfarm():

    if request.method == 'GET':
        return render_template("deepfarm.html", msg='')

    image = request.files['file']    
    NAME = image.filename
    file = open("img.txt",'w')
    file.write(NAME) 
    file.close()
    return render_template("deepfarm.html",msg=image.filename)

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)




# In[8]:


@app.route('/upload',methods=['GET','POST'])
def upload():
    #demo=random.randint(2000, 5000)

    file = open("img.txt","r")
    name=file.read()
    
    file.close()

    source = r'C:\Users\hassa\Music\DeepFarm - Final\deepfarm_front\Test'
    destination = r'C:\Users\hassa\Music\DeepFarm - Final\deepfarm_front\static'

 
    # gather all files
    allfiles = os.listdir(source)
 
    # iterate on all files to move them to destination folder
    for f in allfiles:
        if f==name:
            src_path = os.path.join(source, f)
            dst_path = os.path.join(destination, "OriginalS2.png")
            os.replace(src_path, dst_path)

    with torch.no_grad():
        (x, dates), y = BATCH[name]
        #predictions = utaepaps(x, batch_positions=dates)
        sempred = utae(x, batch_positions=dates)
        sempred = sempred.argmax(dim=1)
        
    plt.matshow(sempred[0].cpu().numpy(),
    cmap=cmap,
    vmin=0,
    vmax=19)

    plt.savefig('./static/Prediction.png')

    plt.imshow(pixel_semantic_annotation[0].squeeze(), cmap=cmap, vmin=0, vmax=20)
    plt.savefig('./static/Label.png')
    
    DIC ={0:"Background",
      1:"Meadow",
      2:"Soft Winter Wheat",3:"Corn",4:"Winter Barley",5:"Winter rapeseed",6:"Spring Barley",7:"Sunflower",8:"Grapevine",9:"Beet",10:"Winter triticale",11:"Winter durum wheat",12:"Fruits,Vegetables,flowers",13:"Potatoes",14:"Leguminous Fodder",15:"Soybeans",16:"Orchard",17:"Mixed Cereal",18:"Sorghum"}


    Picture=sempred[0].numpy()

    Area={}


    for i in range(0,19):

        Area[DIC[i]]=0


    for i in range(0,128):

        for j in range(0,128):


            Area[ DIC[ Picture[i][j] ] ]=Area[ DIC[ Picture[i][j] ] ] + 100
        

    
    return render_template("complete2.html", image_output='OriginalS2.png',image_output1='Label.png',image_output2='Prediction.png',result=Area)


""""
app = Flask(__name__)

@app.route("/upload", methods=['GET','POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print (target)

    if not os.path.isdir(target):
        os.mkdir(target)
    mylist = []
    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, "predicted.jpg"])
        print(destination)
        file.save(destination)
        mylist.append(file)
    print (mylist)

    imgs = [Image.open(i) for i in mylist]
    print(imgs)

    # Find the smallest image, and resize the other images to match it
    min_img_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    img_merge = np.hstack((np.asarray(i.resize(min_img_shape, Image.ANTIALIAS)) for i in imgs))

    # save the horizontally merged images
    img_merge = Image.fromarray(img_merge)

    #image_path= 'output.jpg'
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    #print(dir_path)

    #full_filename = os.path.join('static', 'output.jpg')
    #path = dir_path + full_filename
    #print(full_filename)
    time.sleep(4)
    return render_template("complete.html", image_output='predicted.jpg')
"""
if __name__ == '__main__':
    app.run(debug=True)