# Real time face recognition

Face detection and recognition in a camera input, based on:
* Haar cascade classifier for face detection
* Principal Component Analysis face recognition algorithm

## Testing results

The best results of training and testing on AT&T database obtained for train/test ratio: 80/20 and selection of 4 eigenvectors (corresponding to 4 biggest eigenvalues):

* **96.25% accuracy**

* Training time: 1.45793 s for 320 images, 0.00456 s per image.

* Testing time: 0.39614 s for 80 images, 0.00495 s per image.

* Classification report:

```sh
                     precision        recall      f1-score       support

             s1           1.00          0.90          0.95            10
             s2           1.00          1.00          1.00            10
            s20           0.90          0.90          0.90            10
            s31           0.90          0.90          0.90            10
            s37           1.00          1.00          1.00            10
            s39           1.00          1.00          1.00            10
            s40           0.91          1.00          0.95            10
             s6           1.00          1.00          1.00            10

          micro avg       0.96          0.96          0.96            80
          macro avg       0.96          0.96          0.96            80
       weighted avg       0.96          0.96          0.96            80
```

* Confusion matrix:
```sh
       [[ 9  0  0  0  0  0  1  0]
        [ 0 10  0  0  0  0  0  0]
        [ 0  0  9  1  0  0  0  0]
        [ 0  0  1  9  0  0  0  0]
        [ 0  0  0  0 10  0  0  0]
        [ 0  0  0  0  0 10  0  0]
        [ 0  0  0  0  0  0 10  0]
        [ 0  0  0  0  0  0  0 10]]
 ```

## Getting Started

Training and testing on the prepared dataset - (data.csv)[https://github.com/klaudia-firlag/faceRecognition/blob/master/data.csv] file:
```sh
$ python pca_test.py
```
Training and testing on another dataset:
```sh
$ python pca_test.py -d <database_path>
```

After training and testing recognition is possible with the following command. The recognition is based on camera input and it is initialized with clicking spacebar. ESC invokes quiting the program.
```sh
$ python pca_recognize.py
```

To add people to be recognized, add directories to the existing (people)[https://github.com/klaudia-firlag/faceRecognition/tree/master/people] directory and run:
```sh
$ python add_data.py
```

To delete people from the recognition dataset:
```sh
$ python remove_data.py -t <person_tag>
```
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Installing prerequisites

To install the required packages run the following commands:
```sh
$ pip install numpy
$ pip install scipy
$ pip install -U scikit-learn
$ pip install opencv-python
$ pip install pandas
```

## Author

* **Klaudia Firlag**

## Acknowledgments

* [AT&T database](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) - The Database of Faces was used for training
  and testing the algorithm, Authors: AT&T Laboratories Cambridge

* "Face Recognition Using Principal Component Analysis Method", Authors: Liton Chandra Paul1, Abdulla Al Sumam, [PDF](http://ijarcet.org/wp-content/uploads/IJARCET-VOL-1-ISSUE-9-135-139.pdf)


