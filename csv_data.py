import cv2
import numpy as np
import os
import pandas as pd


class CsvData:
    def __init__(self, data_type):
        self.type = data_type
        self.x = []
        self.y = []

    def save(self, file):
        for idx in range(len(self.x)):
            if self.type == "image_data":
                df = pd.DataFrame({'tag': [self.x[idx]],
                                   'face': [self.y[idx].tolist()]})
            elif self.type == "model_data":
                df = pd.DataFrame({'mean': [self.y.tolist()],
                                   'eigenface': [self.x[idx].tolist()]})

            if idx == 0:
                df.to_csv(path_or_buf=file, index=False)
            else:
                with open(file, 'a') as f:
                    df.to_csv(f, header=False, index=False)

    def read(self, file):
        data = pd.read_csv(file)
        if self.type == "image_data":
            self.x = data["tag"].values
            self.y = np.array([np.fromstring(string[1:-1], dtype=float, sep=',')
                               for string in data["face"].values])
        elif self.type == "model_data":
            self.y = np.fromstring(data["mean"].values[0][1:-1], dtype=float,
                                   sep=', ')
            self.x = np.array(
                [np.fromstring(string[1:-1], dtype=float, sep=', ')
                 for string in data["eigenface"].values])

    def database_to_csv(self, db_path, img_ext, output_file):
        pass


class Model:
    def __init__(self):
        self.mean = []
        self.eigen_vec = []

    def save(self, file):
        for idx in range(len(self.eigen_vec)):
            df = pd.DataFrame(
                {'mean': [self.mean.tolist()],
                 'eigenface': [self.eigen_vec[idx].tolist()]})
            if idx == 0:
                df.to_csv(path_or_buf=file, index=False)
            else:
                with open(file, 'a') as f:
                    df.to_csv(f, header=False, index=False)

    def read(self, file):
        data = pd.read_csv(file)
        self.mean = np.fromstring(data["mean"].values[0][1:-1], dtype=float,
                             sep=', ')
        self.eigen_vec = np.array(
            [np.fromstring(string[1:-1], dtype=float, sep=', ')
             for string in data["eigenface"].values])


def database_to_csv(db_path, img_ext, output_file):
    img_vec_t = []
    tags = []
    first_img_flg = True
    for root, dirs, files in os.walk(db_path):
        for file in files:
            if file.endswith(img_ext):
                # Read tag and image
                tag = root[root.find('/') + 1:]
                tags.append(tag)
                img = cv2.imread(os.path.join(root, file), 0)
                [img_w, img_h] = np.shape(img)
                img_vec = np.reshape(img, img_w * img_h)
                img_vec_t.append(img_vec)

                # Save tag and image vector to a csv file
                df = pd.DataFrame({'tag': [tag], 'face': [img_vec.tolist()]})
                if first_img_flg:
                    df.to_csv(path_or_buf=output_file, index=False)
                else:
                    with open(output_file, 'a') as f:
                        df.to_csv(f, header=False, index=False)

                first_img_flg = False


def read_data(db_path, img_ext, data_file):
    if not os.path.isfile(data_file):
        database_to_csv(db_path, img_ext, data_file)

    data = pd.read_csv(data_file)
    tags = data["tag"].values
    img_vec = np.array([np.fromstring(string[1:-1], dtype=int, sep=',')
                        for string in data["face"].values])

    return np.transpose(img_vec), tags

