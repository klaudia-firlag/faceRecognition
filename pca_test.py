import numpy as np
import optparse
import sys

from csv_data import CsvData, read_data
from pca_train import train
from time import time
from sklearn.metrics import classification_report, confusion_matrix


def disp_results(accuracy, tags, tags_pred, train_time, test_time,
                 train_size, test_size):
    print("Training time: {:.5f} s for {} images.".format(train_time,
                                                           train_size))
    print("               {:.5f} s per image.\n".format(train_time /
                                                         train_size))
    print("Testing time:  {:.5f} s for {} images.".format(test_time,
                                                           test_size))
    print("               {:.5f} s per image.\n".format(test_time/test_size))
    print("Accuracy: {} %.\n".format(accuracy))
    print("Classification report:")
    print(classification_report(tags, tags_pred))
    print("Confusion matrix:")
    print(confusion_matrix(tags, tags_pred))


def test(img_vec, tags, mean, eig_vec, eig_vec_num):
    tested_num = 0
    correct_recognition = 0

    img_idx = 0
    tags_pred = []
    end_idx = len(np.transpose(img_vec))
    while img_idx < end_idx:
        # Weights of test image
        image = np.transpose(img_vec)[img_idx]
        tag = tags[img_idx]
        pca_data = image - mean
        test_weights = []
        for i in range(eig_vec_num):
            test_weights.append(np.dot(eig_vec[i],
                                       np.transpose(pca_data)))

        # Weights of the rest of the data set
        # (without the test image)
        img_vec2 = np.transpose(img_vec)
        data = np.concatenate([img_vec2[:img_idx],
                               img_vec2[img_idx + 1:]], axis=0)
        pca_data = data - mean
        data_weights = np.transpose(np.dot(eig_vec,
                                           np.transpose(pca_data)))

        # Euclidean distance between weights
        dist = []
        for idx in range(len(data_weights)):
            diff = np.linalg.norm(test_weights - data_weights[idx])
            dist.append(diff)
        tested_num += 1
        recognized_tag = tags[np.argmin(dist)]
        tags_pred.append(recognized_tag)
        if recognized_tag == tag:
            correct_recognition += 1
        img_idx += 1

    if 0 == tested_num:
        return 0

    accuracy = 100 * correct_recognition / tested_num

    return accuracy, tags, tags_pred


def split_train_test(train_test_split, in_data, label):
    img_num = len(in_data)
    train_size = int(train_test_split * img_num)
    test_size = img_num - train_size

    # Initialize test and train data variables
    in_data_train = np.zeros((train_size, in_data.shape[1]), dtype=int)
    label_train = [None] * train_size
    in_data_test = np.zeros((test_size, in_data.shape[1]), dtype=int)
    label_test = [None] * test_size

    # Split the dataset
    for idx in range(train_size):
        in_data_train[idx] = in_data[idx]
        label_train[idx] = label[idx]
    for idx in range(test_size):
        in_data_test[idx] = in_data[train_size + idx]
        label_test[idx] = label[train_size + idx]

    return [np.transpose(in_data_train), np.transpose(in_data_test),
            label_train, label_test, train_size, test_size]


def parse_args(parser):
    options, args = parser.parse_args()
    return options


def add_parser():
    parser = optparse.OptionParser()

    parser.add_option('-d', '--database', action="store", dest="db_path",
                      help="path to the image database", default="database")
    parser.add_option('-e', '--fileextension', action="store", dest="img_ext",
                      help="image extension", default="pgm")
    parser.add_option('-f', '--datafile', action="store", dest="data_file",
                      help="csv file with all image data", default="data.csv")
    parser.add_option('-n', '--testdatafile', action="store",
                      dest="test_data_file",
                      help="csv file with test image data",
                      default="data_recog.csv")
    parser.add_option('-m', '--modelfile', action="store",
                      dest="model_file",
                      help="csv file with model parameters",
                      default="pca_model.csv")
    parser.add_option('-s', '--traintestsplit', action="store",
                      dest="train_test_split",
                      help="train/test data split ratio", default="0.8")
    parser.add_option('-v', '--eigenvectors', action="store",
                      dest="eig_vec_num",
                      help="number of eigenvectors to keep", default="4")
    return parser


def test_main(*args, **kwargs):
    parser = add_parser()
    args = parse_args(parser)

    print("Reading data...")
    face_data, tag_data = read_data(args.db_path, args.img_ext,
                                        args.data_file)
    [x_train, x_test, y_train, y_test, train_size, test_size] = \
        split_train_test(float(args.train_test_split),
                         np.transpose(face_data), tag_data)

    print("Training...")
    t0 = time()
    model = CsvData("model_data")
    model.y, model.x = train(x_train, int(args.eig_vec_num))
    train_time = time() - t0
    print("Testing...")
    t0 = time()
    accuracy, tags, tags_pred = test(x_test, y_test, model.y,
                                     model.x, int(args.eig_vec_num))
    test_time = time() - t0

    print("Saving data...\n")
    model.save(args.model_file)
    recog_data = CsvData("image_data")
    recog_data.x = y_test
    recog_data.y = np.transpose(x_test)
    recog_data.save(args.test_data_file)

    disp_results(accuracy, tags, tags_pred, train_time, test_time,
                 train_size, test_size)


if __name__ == "__main__":
    test_main(sys.argv[1:])
