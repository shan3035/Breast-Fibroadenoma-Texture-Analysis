import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib

from PIL import Image

from skimage.feature import (
    graycomatrix,
    graycoprops,
    local_binary_pattern
)



st.set_page_config(
    page_title="Breast Fibroadenoma Texture Analysis",
    layout="wide"
)

st.title("AI-Assisted Breast Fibroadenoma Texture Analysis")


model = joblib.load("breast_model.pkl")

scaler = joblib.load("scaler.pkl")



def extract_features(img):

    img = cv2.resize(img, (256,256))



    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    img = clahe.apply(img)



    img = cv2.GaussianBlur(
        img,
        (3,3),
        0
    )



    glcm = graycomatrix(
        img,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(
        glcm,
        'contrast'
    ).mean()

    correlation = graycoprops(
        glcm,
        'correlation'
    ).mean()

    homogeneity = graycoprops(
        glcm,
        'homogeneity'
    ).mean()

    energy = graycoprops(
        glcm,
        'energy'
    ).mean()

    dissimilarity = graycoprops(
        glcm,
        'dissimilarity'
    ).mean()



    p = glcm[:,:,0,0]

    entropy = -np.sum(
        p * np.log(p + 1e-10)
    )


    mean = np.mean(img)

    variance = np.var(img)


    radius = 1

    points = 8 * radius

    lbp = local_binary_pattern(
        img,
        points,
        radius,
        method='uniform'
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, points + 3),
        range=(0, points + 2)
    )

    hist = hist.astype("float")

    hist /= (hist.sum() + 1e-6)


    features = [
        contrast,
        correlation,
        homogeneity,
        energy,
        dissimilarity,
        entropy,
        mean,
        variance
    ]

    features.extend(hist)

    return features


def detect_roi(img):

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    enhanced = clahe.apply(img)

    blur = cv2.GaussianBlur(
        enhanced,
        (5,5),
        0
    )

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5,5), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:

        return img, (0,0,img.shape[1],img.shape[0])

    valid_contours = []

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if 500 < area < 50000:

            valid_contours.append(cnt)

    if len(valid_contours) == 0:

        return img, (0,0,img.shape[1],img.shape[0])

    best_contour = max(
        valid_contours,
        key=cv2.contourArea
    )

    x,y,w,h = cv2.boundingRect(best_contour)

    padding = 20

    x = max(0, x-padding)

    y = max(0, y-padding)

    w = min(img.shape[1]-x, w+2*padding)

    h = min(img.shape[0]-y, h+2*padding)

    roi = img[y:y+h, x:x+w]

    return roi, (x,y,w,h)

uploaded_file = st.file_uploader(
    "Upload Breast Ultrasound Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

  

    image = Image.open(uploaded_file).convert("L")

    img = np.array(image)

    img = cv2.resize(
        img,
        (512,512)
    )

 

    st.subheader("Original Ultrasound Image")

    st.image(
        img,
        width=300
    )


    roi, (x,y,w,h) = detect_roi(img)

    output = cv2.cvtColor(
        img,
        cv2.COLOR_GRAY2BGR
    )

    cv2.rectangle(
        output,
        (x,y),
        (x+w,y+h),
        (0,255,0),
        2
    )

    st.subheader("Automatic ROI Detection")

    st.image(
        output,
        width=300
    )

  

    roi = cv2.resize(
        roi,
        (256,256)
    )

    features = extract_features(roi)

    features_scaled = scaler.transform([features])

 

    prediction = model.predict(features_scaled)[0]

    probabilities = model.predict_proba(features_scaled)[0]

    classes = model.classes_

    prob_dict = dict(zip(classes, probabilities))

    benign_prob = prob_dict.get('benign', 0) * 100

    malignant_prob = prob_dict.get('malignant', 0) * 100

    

    if malignant_prob > benign_prob:

        final_prediction = "malignant"

        confidence = malignant_prob

    else:

        final_prediction = "benign"

        confidence = benign_prob

    

    contrast = features[0]

    entropy = features[5]

    dissimilarity = features[4]

    variance = features[7]

    heterogeneity_score = (
        0.3 * contrast +
        0.3 * entropy +
        0.2 * dissimilarity +
        0.2 * variance
    )

   

    st.subheader("Prediction Result")

    if final_prediction == "malignant":

        st.error(
            f"Prediction: {final_prediction.upper()}"
        )

    else:

        st.success(
            f"Prediction: {final_prediction.upper()}"
        )

    st.write(
        f"Confidence Score: {confidence:.2f}%"
    )

  

    st.subheader("Prediction Probability")

    fig1, ax1 = plt.subplots()

    class_names = [
        "benign",
        "malignant"
    ]

    prob_values = [
        benign_prob,
        malignant_prob
    ]

    ax1.bar(
        class_names,
        prob_values
    )

    ax1.set_ylabel("Probability (%)")

    st.pyplot(fig1)

   

    st.subheader("Heterogeneity Analysis")

    st.write(
        f"Heterogeneity Score: {heterogeneity_score:.2f}"
    )

    if heterogeneity_score < 20:

        st.success(
            "Low heterogeneity detected"
        )

    elif heterogeneity_score < 50:

        st.warning(
            "Moderate heterogeneity detected"
        )

    else:

        st.error(
            "High heterogeneity detected"
        )

   

    st.subheader("Texture Feature Analysis")

    feature_names = [
        "contrast",
        "correlation",
        "homogeneity",
        "energy",
        "dissimilarity",
        "entropy",
        "mean",
        "variance"
    ]

    texture_values = features[:8]

    fig2, ax2 = plt.subplots(
        figsize=(10,5)
    )

    ax2.bar(
        feature_names,
        texture_values
    )

    plt.xticks(rotation=45)

    ax2.set_ylabel("Feature Value")

    st.pyplot(fig2)


st.markdown("---")

st.markdown(
    "AI-Assisted Medical Image Processing "
    "Framework for Breast Ultrasound Analysis"
)