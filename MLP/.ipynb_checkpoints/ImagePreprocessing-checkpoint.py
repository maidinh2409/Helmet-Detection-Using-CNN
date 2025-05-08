from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt 
import xml.etree.ElementTree as et
import cv2
import os

class ImgObject:
    def __init__(self, img_path, label_path):
        self.img_path = img_path
        self.label_path = label_path

    def show_images_inline(self, images):
        images = images[:5]  
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))

        if n == 1:
            axes = [axes]

        for i, img in enumerate(images):
            ax = axes[i]
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis('off')
    
        plt.tight_layout()
        plt.show()

    # Show loaded image
    def show_image(self):
        img = Image.open(self.img_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def img_overview(self):
        img = cv2.imread(self.img_path)
        height, width, channels = img.shape
        file = os.path.basename(self.img_path)
        file_name = os.path.splitext(file)[0]
        
        return file_name, height, width, channels

    def obj_overview(self):
        tree = et.parse(self.label_path)
        root = tree.getroot()
        object_list = []

        file_name = root.find("filename").text.strip()

        for i, obj in enumerate(root.findall('object')):
            class_name = obj.find("name").text.strip()

            if class_name == "Without Helmet":
                class_name = 1
            elif class_name == "With Helmet":
                class_name = 0
            else:
                class_name = -1
                
            bbox = obj.find("bndbox")

            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            object_list.append((f"{file_name}_{i}", class_name, xmin, ymin, xmax, ymax))
        return object_list

    # Crop objects from parent image into images
    def crop_object(self):
        img = cv2.imread(self.img_path)
        file_name, height, width, channels = self.img_overview()
        object_list = self.obj_overview()
        images = []
        labels = []
        
        for obj in object_list:
            print(obj)
            print(type(obj))
            _, class_id, xmin, ymin, xmax, ymax = obj
            cropped = img[ymin:ymax, xmin:xmax]

            cropped_resized = cv2.resize(cropped, (256, 256))
    
            images.append(cropped_resized)
            labels.append(class_id)
        return images, labels
        
        

                
        