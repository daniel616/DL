import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import torchvision
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from cuda_handler import device

class GradModel(nn.Module):
    def __init__(self,model):
        super(GradModel,self).__init__()
        self.model=model
        try:
            self.last_conv=model.last_conv
            self.remaining_layers=model.remaining_layers
        except AttributeError:
            self.last_conv=model[:-2]
            self.remaining_layers=lambda x: model[-1](model[-2](x))

    def setgrad(self,grad):
        self.grad=grad

    def forward(self,x):
        x=self.last_conv(x)
        x.register_hook(self.setgrad)
        x=self.remaining_layers(x)
        return x

def show_images(img_gen,out_dir="./gen_images/",with_label=True):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.dirname(out_dir))

    for idx, sample in enumerate(img_gen):
        if with_label:
            fig=plt.figure()
            image, box= sample['image'], to_numpy(sample['box'])
            label=sample['label'].item()
            labelstr= "A" if label == 1 else "N"
            z=str(sample['z'])
            for j in range(image.shape[0]):
                window = image[j].reshape(512, 512)
                window=to_numpy(window)
                ax = fig.add_subplot(1,image.shape[1],j+1)
                plt.imshow(window,cmap="Greys_r")

                if int(box[0])!=-1:
                    width=box[2]-box[0]
                    height=box[3]-box[1]
                    rect = patches.Rectangle((box[0],box[3]), width, height, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)

            fig.set_size_inches(np.array(fig.get_size_inches()) * image.shape[0])
            plt.savefig(out_dir+str(idx)+labelstr+"_"+z+".png")
            plt.close(fig)


        else:
            #buggy if more than one channel
            image=sample.reshape(sample.shape[0],1,512,512)
            torchvision.utils.save_image(image, out_dir + str(idx)+ '.png')



def show_z(img_gen,out_dir="./gen_images/",with_label=True):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.dirname(out_dir))
    for idx, sample in enumerate(img_gen):
        if with_label:
            image, box= sample['image'], sample['box']
            label=sample['label'].item()
            labelstr= "A" if label == 1 else "N"
            z=str(sample['z'])
            image=image[0].reshape(512,512)
            image=to_numpy(image)
            box=to_numpy(box)
            ax=plt.gca()
            #import pdb; pdb.set_trace()
            plt.imshow(image,cmap="Greys_r")


            if int(box[0])!=-1:
                width=box[2]-box[0]
                height=box[3]-box[1]
                rect = patches.Rectangle((box[0],box[3]), width, height, linewidth=1, edgecolor='r', facecolor='none')

                ax.add_patch(rect)

            plt.savefig(out_dir+z+".png")
            ax.clear()

        else:
            image=sample.reshape(sample.shape[0],1,512,512)
            torchvision.utils.save_image(image, out_dir + str(idx)+ '.png')

def examine_ssbr(img_gen,model,out_dir="./ssbr_preds/"):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.dirname(out_dir))
    for idx,sample in enumerate(img_gen):
        sample=sample.to(device)
        preds=to_numpy(model(sample).reshape(-1))
        preds=preds.tolist()
        images=[]
        for i in range(sample.shape[0]):
            img=sample[i][0].reshape(512,512)
            img= to_numpy(img)
            images.append(img)
        xshow_images(images,titles=preds)
        plt.savefig(out_dir+str(idx)+".png")
        plt.close()


#https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
def xshow_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        #a = fig.add_subplot(n+1,np.ceil(n_images / float(cols)),cols)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    #plt.show()


#Assumes batch dimension is 1
def gradcam(generator,model,out_dir="./gradcam/"):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    n_model=GradModel(model)
    for idx,sample in enumerate(generator):
        label=sample['label'].item()
        suf="A" if label==1 else "N"

        pic=sample['image']
        pic=pic.to(device)
        pic=pic.unsqueeze(0)
        out=n_model(pic)

        out=out.squeeze()
        out.backward()
        pred= "_P:A" if out>0.5 else "_P:N"
        gradients=n_model.grad

        map_weights=torch.mean(gradients,dim=[2,3]).squeeze(dim=0)

        activations=n_model.last_conv(pic).squeeze(dim=0)

        for idx2, val in enumerate(map_weights):
            activations[idx2,:,:]*=val

        heatmap=torch.mean(activations,dim=0)
        heatmap=heatmap.clamp(min=0)

        heatmap/=0.001
        heatmap,pic=to_numpy(heatmap),to_numpy(pic)[0,0,:,:]


        heatmap,pic=scale_255(heatmap),scale_255(pic)
        heatmap = cv2.resize(heatmap, (pic.shape[0], pic.shape[1]))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + np.expand_dims(pic,-1)

        box=to_numpy(sample['box'])
        cv2.rectangle(superimposed_img,(box[0],box[1]),(box[1],box[2]),(0,255,0),2)
        cv2.imwrite(out_dir+str(idx)+suf+pred+".png", superimposed_img)


def scale_255(pic):
    if isinstance(pic,np.ndarray):
        min,max=np.min(pic),np.max(pic)
        pic=255*(pic-min)/(max-min)
        return pic.astype('uint8')
    if isinstance(pic,torch.Tensor):
        min,max=torch.max(pic),torch.min(pic)
        pic=255*(pic-min)/(max-min)
        return pic.type(torch.uint8)

    raise TypeError("pic type unsupported")


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()
