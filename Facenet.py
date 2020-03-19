import keras as k, numpy as np
import cv2

def load_model(path="models\\facenet_keras.h5"):
    """ 
    Takes path where model is stored and returns the loaded model
    path : Path to the facenet model
    """
    model = None
    try:
        model = k.models.load_model(path, compile=False)
        print("[INFO] Facenet model loaded")
    except Exception as e:
        print(f"[ERROR] Error occured while Loading the {path} model : {e}")
    return model


def preprocess(f):
    """
    Preprocesses the face image and returns the processed image
    f : Face image
    """
    sf = None
    try:
        sf = cv2.resize(f, (160,160))
        sf = sf.astype("float32")
        mean, std = sf.mean(), sf.std()
        sf = (sf - mean) / std
    except Exception as e:
        print(f"[ERROR] Facenet preprocess :  {e}")
    return sf
    
    
def l2_normalize(x, axis=-1, epsilon=1e-10):
    """
    Normalizes the facenet features
    x : (1,128) dimensional 
    """
    output = None
    try:
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    except Exception as e:
        print(f"[ERROR] Facenet l2_normalize : {e}")
    return output

def l2_normalize_div(x, axis=-1, epsilon=1e-10):
    output = None
    try:

        x_cp = x.copy()
        print("original : ", x)
        x = np.square(x)
        print("Square : ", x)
        x = np.sum(x, axis=axis, keepdims=True)
        print("Sum : ", x)
        x = np.maximum(x, epsilon)
        print("Max : ", x)
        x = np.sqrt(x)
        print("Sqrt : ", x)
        output = x_cp / x
        print("output", output)
    except Exception as e:
        print(f"[ERROR] Facenet l2_normalize : {e}")
    return output
    
def get_embeddings(faces, model, verbose=0):
    """
    Accepts face image and returns 128 dimensional feature vector
    face_img : Face image
    model : facenet model
    """
    face_features = []
    try :
        for ix,face in enumerate(faces):
            s_face = preprocess(face)
            s_face = s_face.reshape((-1,160,160,3))
            feature = model.predict(s_face)
            nfeature = l2_normalize(feature)
            face_features.append(nfeature[0])
            if verbose == 1 : print(f"Processed : {ix+1}/{len(faces)}");
    except Exception as e:
        print(f"[ERROR] Facenet get_embeddings : {e}")
    
    return np.asarray(face_features)
    
    
if __name__ == "__main__":
    fn = load_model()
    tt = cv2.imread("1.jpg")
    print("------------------------------------------------")
    print(get_embeddings([tt], fn))