from FaceDetectionSSD import FaceDetectionSSD
from Facenet1 import Facenet
from augmentation import augmentImage
from datetime import datetime
import utils as u
import cv2
import numpy as np


def main(videoSource: str, imageSource: str):
    camIndex = 0
    fd = FaceDetectionSSD()
    facenet = Facenet()
    data = []

    # try:
    #     cam_arr = u.getNumberOfCameras()
    #     print(f"[ INFO ]Number of cameras detected : {len(cam_arr)}")
    #     if len(cam_arr) > 1:
    #         print("inside")
    #         camIndex = 1
    # except Exception as e:
    #     print(f"[ Error ] :`` {e}")
    #     camIndex = 0

    img = cv2.imread(imageSource)
    face_img, source  = None, None
    if img is None:
        print("[ERROR] Image Not Found....")
        return
    else:

        face_imgs = []
        print("[INFO] Augmenting Input Image")
        augmented_images = augmentImage(img)

        print("[INFO] Extracting faces from augmented images")
        for ix, img in enumerate(augmented_images):
            face_locations = fd.detect_faces(img)
            if len(face_locations) > 0:
                face_img = fd.extract_faces(img, [face_locations[0]])[0]
                face_imgs.append(face_img)

        if len(face_imgs) > 0:
            # cv2.imshow(f"image", face_imgs[0])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            print("[INFO] Generating Embeddings : ")
            sources = facenet.get_embeddings(face_imgs, verbose=1)
        else:
            print("[ERROR] No Face Found....")

        # face_locations = fd.detect_faces(img)
        # if len(face_locations) > 0:
        #     face_img = fd.extract_faces(img, [face_locations[0]])[0]
        #     augmented_images = augmentImage(face_img)
        #     # source = facenet.get_embeddings([face_img])
        #     sources = facenet.get_embeddings(augmented_images, verbose=1)
        #     # print(source)
        #     cv2.imshow("image", face_img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

    cap = cv2.VideoCapture(videoSource)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    FPM = FPS / 1000
    TOTAL_FRAMES = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    count, frames = 0, []
    print(f"FPM : {FPM}, FPS : {FPS}, TOTAL FRAMES : {TOTAL_FRAMES}, Seconds : {TOTAL_FRAMES/FPS}")

    while True:
        ret, frame = cap.read()

        if ret:
            count += 1
            face_locations = fd.detect_faces(frame)
            if len(face_locations) > 0:
                detected_faces = fd.extract_faces(frame, face_locations)
                face_features = facenet.get_embeddings(detected_faces)

                if face_features is None:
                    continue

                predictions = u.getPrediction(sources, face_features, 0.7)

                milSeconds = int(count / FPM)
                time_ = u.convertMilSeconds(milSeconds)
                print(time_, predictions)

                if len(predictions) > 0:
                    data.append(f"{time_}   -   {predictions[0]}\n")

                frame = u.draw_predictions(frame, face_locations, predictions)

            frames.append(frame)

            # cv2.imshow("video", frame)
            # if cv2.waitKey(1) == 13:
            #     break
        else:
            break
    
    if len(data) > 0:
        dt = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        # u.write_data(data, f"logs\\log-{dt}.txt")
        u.writeVideo("logs\\output_vid2.mp4", frames, FPS)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("data\\vid2.MP4", "data\\2.JPG")
