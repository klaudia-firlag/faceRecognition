import cv2
import numpy as np
import optparse
import sys

from csv_data import CsvData, read_data
from detection import detect_faces


def recognize(img_vec, tags, mean, eig_vec):
    face_classifier_name = 'haarcascade_frontalface_default.xml'
    face_classifier = cv2.CascadeClassifier(face_classifier_name)

    # Capture video
    frame_first_flg = 1
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_vec = detect_faces(gray, face_classifier)
        cv2.imshow("frame", gray)

        key = cv2.waitKey(1)
        if key & 0xFF == ord(' '):
            if not faces_vec:
                print("No faces detected. Try again.")
            else:
                # Weights of the dataset
                pca_data = img_vec - mean
                data_weights = np.transpose(np.dot(eig_vec,
                                                   np.transpose(pca_data)))

                img_idx = 0
                for face in faces_vec:
                    # Weights of test image
                    pca_data = face - mean
                    test_weights = []
                    for i in range(len(eig_vec)):
                        test_weights.append(np.dot(eig_vec[i],
                                                   np.transpose(pca_data)))

                    # Euclidean distance between weights
                    dist = []
                    for idx in range(len(data_weights)):
                        diff = np.linalg.norm(test_weights - data_weights[idx])
                        dist.append(diff)
                    min_dist = np.argmin(dist)
                    recognized_tag = tags[np.argmin(dist)]
                    img_idx += 1

                    print("Recognized person #{}: {}".format(img_idx,
                                                             recognized_tag))
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args(parser):
    options, args = parser.parse_args()
    return options


def add_parser():
    parser = optparse.OptionParser()
    parser.add_option('-f', '--datafile', action="store", dest="data_file",
                      help="csv file with all image data",
                      default="data_personal.csv")
    parser.add_option('-m', '--modelfile', action="store", dest="model_file",
                      help="csv file with model data", default="model.csv")
    return parser


def recognition_main(*args, **kwargs):
    print("Parsing data...")
    parser = add_parser()
    args = parse_args(parser)

    print("Reading data...")
    data = CsvData("image_data")
    model = CsvData("model_data")
    data.read(args.data_file)
    model.read(args.model_file)

    recognize(data.y, data.x, model.y, model.x)


if __name__ == "__main__":
    recognition_main(sys.argv[1:])
