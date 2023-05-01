from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import os
import pandas as pd
import torchvision.transforms as tvt
from torchvision.io import read_image
import argparse

# Global Variables
train_directory = r"/Users/nikitaravi/Documents/Academics/ECE 60146/HW5/train2014"
val_directory = r"/Users/nikitaravi/Documents/Academics/ECE 60146/HW5/val2014"
train_annotations = r"/Users/nikitaravi/Documents/Academics/ECE 60146/HW5/annotations/instances_train2014.json"
val_annotations = r"/Users/nikitaravi/Documents/Academics/ECE 60146/HW5/annotations/instances_val2014.json"
coco_train = COCO(train_annotations)
coco_val = COCO(val_annotations)
categories = ["pizza", "bus", "cat"]
columns = ["id", "category", "path_to_image", "x1", "y1", "x2", "y2"]
new_image_size = 256
resize_image = tvt.Compose([tvt.Resize((new_image_size,new_image_size))])


def parser():
    parser = argparse.ArgumentParser(description="Load data into dataframe")
    parser.add_argument("--train", action="store_true", help="Download training data or testing data")
    args = parser.parse_args()
    return args

def create_dataframe(train=True):
    saved_count = 0
    df = pd.DataFrame(columns=columns)
    ids, class_list, paths_to_image, bbox_x1s, bbox_y1s, bbox_x2s, bbox_y2s = [], [], [], [], [], [], []
    directory = train_directory if train else val_directory
    new_dirname = f"./train" if train else f"./val"
    coco = coco_train if train else coco_val
    os.makedirs(new_dirname, exist_ok=True)
    for class_idx, category in enumerate(categories):
        category_id = coco.getCatIds(catNms=category)
        image_ids = coco.getImgIds(catIds=category_id)
        for img_id in image_ids:
            current_image = coco.loadImgs(ids=img_id)
            current_id = current_image[0]["id"] 
            annotation_id = coco.getAnnIds(imgIds=img_id, catIds=category_id, iscrowd=False)
            annotation = coco.loadAnns(ids=annotation_id)

            check_bbox = False
            if(annotation[0]["area"] > 200 * 200):
                saved_count += 1
                check_bbox = True
            if(check_bbox):
                path_to_image = current_image[0]["file_name"]
                original_width, original_height = current_image[0]["width"], current_image[0]["height"]

                # Resize Image
                original_image = resize_image(read_image(os.path.join(directory, path_to_image)))
                original_image = original_image.repeat(3,1,1) if original_image.size()[0] == 1 else original_image # Check if image has three channels
                original_image = tvt.functional.to_pil_image(original_image).convert("RGB") # convert to RGB image

                # Resize bbox
                annotation_width, annotation_height = annotation[0]["bbox"][2], annotation[0]["bbox"][3]
                annotation_left_x, annotation_left_y = annotation[0]["bbox"][0], annotation[0]["bbox"][1]

                x_scale = new_image_size / original_width
                y_scale = new_image_size / original_height

                new_bbox_width = x_scale * annotation_width
                new_bbox_height = y_scale * annotation_height
                new_x1 = x_scale * annotation_left_x
                new_y1 = y_scale * annotation_left_y

                new_x2 = new_x1 + new_bbox_width
                new_y2 = new_y1 + new_bbox_height

                image_filename = f"{category}_{current_id}.png"
                original_image.save(os.path.join(new_dirname, image_filename))
                
                # Add everything to the list to add to the df
                ids.append(current_id)
                class_list.append(category)
                paths_to_image.append(os.path.join(new_dirname, image_filename))
                bbox_x1s.append(new_x1)
                bbox_y1s.append(new_x2)
                bbox_x2s.append(new_y1)
                bbox_y2s.append(new_y2)

    df["id"] = ids
    df["category"] = class_list
    df["path_to_image"] = paths_to_image
    df["x1"] = bbox_x1s
    df["x2"] = bbox_y1s
    df["y1"] = bbox_x2s
    df["y2"] = bbox_y2s

    filename = "train_data.csv" if train else "test_data.csv"
    df.to_csv(filename)


if __name__ == "__main__":
    args = parser()
    train = True if args.train else False

    create_dataframe(train)
