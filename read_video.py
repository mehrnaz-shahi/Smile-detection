import cv2
from functions import hog_feature, lbp_feature, concatenate, detect_lip_test, load_svm, detect_faces

cap = cv2.VideoCapture('test4.mp4')
facedetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
vwriter = cv2.VideoWriter('out.wmv', cv2.VideoWriter_fourcc(*'WMV1'), 20, (640, 480))
clf = load_svm()

while (True):
    ret, frame = cap.read()
    if ret:

        faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 10, 10), 2)

            r = detect_lip_test(frame, faces, face_img)

            hog_features = hog_feature(r)
            #  LBP
            lbp_features = lbp_feature(r)

            # Combine  HOG & LBP
            features = concatenate(hog_features, lbp_features)

            label = clf.predict([features])[0]

            if label == 1:
                image = cv2.putText(frame, 'Smiling', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                image = cv2.putText(frame, "Don't Smiling", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2, cv2.LINE_AA)

        vwriter.write(frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
    else:
        vwriter.release()

cap.release()
