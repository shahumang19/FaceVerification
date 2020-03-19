from FaceDetectionSSD import FaceDetectionSSD
from Facenet1 import Facenet
from utils import draw_predictions, getNumberOfCameras, getDistance, getPrediction
import cv2
import numpy as np


def main(videoSource: str, imageSource: str):
    camIndex = 0
    fd = FaceDetectionSSD()
    facenet = Facenet()

    # try:
    #     cam_arr = getNumberOfCameras()
    #     print(f"[ INFO ]Number of cameras detected : {len(cam_arr)}")
    #     if len(cam_arr) > 1:
    #         print("inside")
    #         camIndex = 1
    # except Exception as e:
    #     print(f"[ Error ] : {e}")
    #     camIndex = 0

    img = cv2.imread(imageSource)
    face_img, source  = None, None
    if img is None:
        print("[ERROR] Image Not Found....")
        return
    else:
        face_locations = fd.detect_faces(img)
        if len(face_locations) > 0:
            face_img = fd.extract_faces(img, [face_locations[0]])[0]
            source = facenet.get_embeddings(face_img)[0]
            cv2.imshow("image", face_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    cap = cv2.VideoCapture(videoSource)

    while True:
        ret, frame = cap.read()

        if ret:
            face_locations = fd.detect_faces(frame)
            if len(face_locations) > 0:
                detected_faces = fd.extract_faces(frame, face_locations)
                face_features = facenet.get_embeddings(detected_faces)

                predictions = getPrediction(source, face_features)
                frame = draw_predictions(frame, face_locations, predictions)

            cv2.imshow("video", frame)
            if cv2.waitKey(1) == 13:
                break
        else:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("data\\vid.MP4", "data\\2.JPG")
