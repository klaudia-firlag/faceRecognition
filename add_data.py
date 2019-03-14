import cv2
import numpy as np
import optparse
import os
import pandas as pd
import sys

from csv_data import CsvData
from detection import detect_faces
from shutil import copyfile


def parse_args(parser):
    options, args = parser.parse_args()
    return options


def add_parser():
    parser = optparse.OptionParser()

    parser.add_option('-o', '--olddatafile', action="store",
                      dest="old_data_file",
                      help="existing csv file with image data from a database",
                      default="data_recog.csv")
    parser.add_option('-n', '--newdatafile', action="store",
                      dest="new_data_file",
                      help="new csv file with all image data",
                      default="data_personal.csv")
    parser.add_option('-a', '--addpeople', action="store", dest="new_db_path",
                      help="path to a folder with photos of 1 person",
                      default="people")
    return parser


def add_people(args):
    print("Parsing data...")
    parser = add_parser()
    args = parse_args(parser)

    print("Reading data...")
    if not os.path.isfile(args.new_data_file):
        copyfile(args.old_data_file, args.new_data_file)

    data = CsvData("image_data")
    data.read(args.new_data_file)

    print("Adding images to the dataset...")
    face_classifier_name = 'haarcascade_frontalface_default.xml'
    face_classifier = cv2.CascadeClassifier(face_classifier_name)
    tag_prev = ''
    for root, dir, files in os.walk(args.new_db_path):
        for file in files:
            tag = root[root.rfind('/') + 1:]
            if tag != tag_prev:
                tag_prev = tag
                if tag in data.x:
                    print("{} already in the database. "
                          "Omitting...".format(tag))
                    continue
                print("Adding {}...".format(tag))
            img = cv2.imread(os.path.join(root, file), 0)
            img = detect_faces(img, face_classifier, detect_one=True)
            [img_w, img_h] = np.shape(img)
            img_vec = np.reshape(img, img_w * img_h)

            # Save tag and image vector to a csv file
            df = pd.DataFrame({'tag': [tag], 'face': [img_vec.tolist()]})
            with open(args.new_data_file, 'a') as f:
                df.to_csv(f, header=False, index=False)

    print("Done.")


if __name__ == "__main__":
    add_people(sys.argv[1:])
