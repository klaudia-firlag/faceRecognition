import cv2


def detect_faces(img, face_cascade, detect_one=False):
    faces_points = face_cascade.detectMultiScale(img, 1.3, 5)
    if faces_points == ():
        return ()

    faces_vec = []
    face_idx = 0
    for (x, y, w, h) in faces_points:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = img[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (112, 92))
        if detect_one:
            return face_resized
        else:
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, "face #{}".format(face_idx + 1), (x + 5, y + 15),
                        font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, "face #{}".format(face_idx + 1), (x + 5, y + 15),
                        font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        face_idx += 1

        faces_vec.append((face_resized.reshape(112 * 92)))
    return faces_vec
