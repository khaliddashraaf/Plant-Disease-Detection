import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = r"C:\Users\Khalid\Desktop\GUC\NTI\Plant_img_segmentation\data\model"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")



class CustomConfig(Config):
# """Configuration for training on the custom  dataset.
# Derives from the base Config class and overrides some values.
# """
    # Give the configuration a recognizable name
    NAME = "leaf"

    # We use a GPU with 24GB memory, which can fit 4 images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + Apple_Black_Rot, Apple_Healthy, Apple_Rust, Apple_Scab, Blueberry_Healthy, Corn_Common_Rust, Corn_Gray_Leaf_Spot,Corn_Healthy, Grape_Black_Measles, Grape_Black_Rot, Grape_Healthy, Peach_Bacterial_Spot, Peach_Healthy, Pepper_Bell_Bacterial_Spot, Pepper_Bell_Healthy, Pepper_Early_Blight, Potato_Healthy, Potato_Late_Blight, Rasberry_Healthy, Soybean_Healthy,Strawberry_Leaf_Scroch,Strawberry_Healthy,Tomato_Bacterial_Spot,Tomato_Early_Blight, Tomato_Healthy,Tomato_Late_Blight,Tomato_Leaf_Spot,Tomato_Target_Spot


    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
    
    
        # Add classes. We have only one class to add.
        self.add_class("object", 1, "potato_early_blight")   #potato_Early_Blight
        self.add_class("object", 2, "potato_healthy")
        self.add_class("object", 3, "potato_late_blight")

        
        # Train or validation dataset?
        assert subset in ["train", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        # We mostly care about the x and y coordinates of each region
        
        #annotations1 = json.load(open('D:/python3.6.8_tensorflow_1.14_env/maskrcnn_leave_disease_detection/dataset/train/train.json'))
        annotations1 = json.load(open(os.path.join(dataset_dir, 'potato.json')))
        #keep the name of the json files in the both train and val folders
        
        # print(annotations1)
        annotations = list(annotations1.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        
        # Add images
        for a in annotations:
            # print(a)
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']] 
            objects = [s['region_attributes']['disease'] for s in a['regions']]
            print("objects:",objects)
            name_dict = {"potato_early_blight":1,"potato_healthy":2,"potato_late_blight":3}
            #name_dict = {"Horse": 1,"Man": 2} #,"xyz": 3}
            # key = tuple(name_dict)
            num_ids = [name_dict[a] for a in objects]
        
            # num_ids = [int(n['Event']) for n in objects]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            print("numids",num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
                )

    def load_mask(self, image_id):
    # """Generate instance masks for an image.
    # Returns:
    # masks: A bool array of shape [height, width, instance count] with
    #     one mask per instance.
    # class_ids: a 1D array of class IDs of the instance masks.
    # """
    # If not a Dog-Cat dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids #np.ones([mask.shape[-1]], dtype=np.int32)

def image_reference(self, image_id):
    """Return the path of the image."""
    info = self.image_info[image_id]
    if info["source"] == "object":
        return info["path"]
    else:
        super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(r"C:\Users\Khalid\Desktop\GUC\NTI\Plant_img_segmentation\data\model\dataset", "train")

    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(r"C:\Users\Khalid\Desktop\GUC\NTI\Plant_img_segmentation\data\model\dataset", "test")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='heads')
                
                
                
config = CustomConfig()
model = modellib.MaskRCNN(mode="training", config=config,
                                model_dir=DEFAULT_LOGS_DIR)

weights_path = COCO_WEIGHTS_PATH
    # Download weights file
if not os.path.exists(weights_path):
    utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])

train(model)			