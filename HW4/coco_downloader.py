# Import Libraries
from PIL import Image
from pycocotools.coco import COCO
import os
import requests

class COCODownloader:
    def __init__(self, root_path, coco_json_path, class_names, num_images_per_class):
        self.root_path = root_path
        self.coco_json_path = coco_json_path
        self.class_names = class_names
        self.num_images_per_class = num_images_per_class
        self.coco = COCO(self.coco_json_path)

    def __create_directories(self, class_name):
        path_to_class = os.path.join(self.root_path, class_name)
        if(not os.path.exists(path_to_class)):
            os.makedirs(path_to_class)

        return path_to_class
                
    def resize_image(self, image_path):
        try:
            image = Image.open(image_path)
            if(image.mode != "RGB"):
                image = image.convert(mode="RGB")

            resized_image = image.resize((64, 64), Image.BOX)
            resized_image.save(image_path)
            return True

        except Exception as e:
            print(e)

    def download_coco_images(self):
        for class_name in self.class_names:
            path_to_class = self.__create_directories(class_name=class_name)
            category_ids = self.coco.getCatIds(catNms=class_name)
            image_ids = self.coco.getImgIds(catIds=category_ids)

            images_from_class = self.coco.loadImgs(ids=image_ids)
            count = 0
            for image in images_from_class:
                try:
                    if(count < self.num_images_per_class):
                        image_data = requests.get(image["coco_url"])
                        path_to_image = os.path.join(path_to_class, image["file_name"])
                        with open(path_to_image, "wb") as fptr:
                            fptr.write(image_data.content)
                        if(self.resize_image(path_to_image)):
                            count += 1

                except Exception as e:
                    print(e)

def train():
    root_path = r"/Users/nikitaravi/Documents/Academics/ECE 60146/HW4/Train"
    coco_json_path = r"/Users/nikitaravi/Documents/Academics/ECE 60146/HW4/annotations/instances_train2014.json"
    class_names = ["airplane", "bus", "cat", "dog", "pizza"]
    train_images_per_class = 1500

    coco_downloader = COCODownloader(root_path, coco_json_path, class_names, train_images_per_class)
    coco_downloader.download_coco_images()
    

def val():
    root_path = r"/Users/nikitaravi/Documents/Academics/ECE 60146/HW4/Val"
    coco_json_path = r"/Users/nikitaravi/Documents/Academics/ECE 60146/HW4/annotations/instances_val2014.json"
    class_names = ["airplane", "bus", "cat", "dog", "pizza"]
    val_images_per_class = 500

    coco_downloader = COCODownloader(root_path, coco_json_path, class_names, val_images_per_class)
    coco_downloader.download_coco_images()


if __name__ == "__main__":
    mode = "val"
    if(mode == "train"):
        train()
    elif(mode == "val"):
        val()