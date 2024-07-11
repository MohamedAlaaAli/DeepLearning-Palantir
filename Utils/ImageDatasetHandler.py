import os
import torch
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.classes = os.listdir(data_dir)
        self.class_to_index = {class_ : index for class_, index in enumerate(self.classes)}
        self.transform = transform
        self.img_paths = self.get_img_paths()
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path, label = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            self.transform(img).unsqueeze(0).to("cuda")
        return img
    
    def get_img_paths(self):
        paths = []
        for class_name in self.classes:
            class_dir = os.path.joint(self.data_dir, class_name)
            for img in class_dir:
                img_path = os.path.join(class_dir, img)
                label = self.class_to_index[class_name]
                paths.append((img_path, label))
        return paths