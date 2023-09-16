import numpy as np
import os
import pandas as pd

from sklearn.model_selection import KFold
from PIL import Image


class FeynmanDiagramsModel:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.diagrams_df = pd.read_csv(self.csv_file)

    def data_preprocessing(self):
        # Get images in the right format
        diagrams_code = self.diagrams_df.iloc[:, 0].astype(str)
        diagrams_code = [f"{value}.png" for value in diagrams_code]

        self.diagrams_imgs = []

        for code in diagrams_code:
            image_path = os.path.join("Diagrams_images", code)
            image = self.preprocess_image(image_path)
            self.diagrams_imgs.append(image)

        # Get image description
        self.diagrams_description = self.diagrams_df.iloc[:, 1].astype(str)

    def data_split(self):
        n_splits = 5

        kf = KFold(n_splits, shuffle=True, random_state=42)

        for train_index, test_index in kf.split(self.diagrams_imgs):
            self.X_train, self.X_test = (
                self.diagrams_imgs[train_index],
                self.diagrams_imgs[test_index],
            )
            self.y_train, self.y_test = (
                self.diagrams_description[train_index],
                self.diagrams_description[test_index],
            )

    @staticmethod
    def preprocess_image(img_path, target_size=(224, 224)):
        img = Image.open(img_path)
        img = img.resize(target_size)
        img = np.asarray(img, dtype=np.float32) / 255.0

        return img


### Example ###

# csv = "C:/Projetos/Personal-projects/Feynman-diagrams/CSV_file/feynman_diagrams.csv"

# a = FeynmanDiagramsModel(csv).data_preprocessing()
