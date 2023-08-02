import cv2
from fer import FER
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st
import time

#new_title2 = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">EMOTION DETECTOR</p>'
#st.markdown(new_title2, unsafe_allow_html=True)
st.header('EMOTION DETECTOR')
if st.button('Take photo'):
    #new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 42px;">RAW IMAGE</p>'
    #st.markdown(new_title, unsafe_allow_html=True)
    
     
    cam = cv2.VideoCapture(0)

    


    ret, frame = cam.read()
    


    emotion_detector = FER()
    # Output image's information
    print(emotion_detector.detect_emotions(frame))
    cv2.imshow('frame', frame)
    st.subheader('RAW IMAGE')
    st.image(frame)
    
    
    

    result = emotion_detector.detect_emotions(frame)
    bounding_box = result[0]["box"]
    emotions = result[0]["emotions"]
    cv2.rectangle(frame,(bounding_box[0], bounding_box[1]),(bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),(0, 155, 255), 2,)
    emotion_name, score = emotion_detector.top_emotion(frame )
    
 
    
    
        
    #new_title1 = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">PREDICTED IMAGE</p>'
    #st.markdown(new_title1, unsafe_allow_html=True)
    st.subheader('PREDICTED IMAGE')
    
       #cv2.putText(frame,emotion_score,(bounding_box[0], bounding_box[1] + bounding_box[3] + 30 + index * 15),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA,)
    #Save the result in new image file
    cv2.imwrite("emotion.jpg", frame)
    result_image = mpimg.imread('emotion.jpg')
    imgplot = plt.imshow(result_image)
    # Display Output Image
    my_bar = st.progress(0)



    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
        
    st.balloons() 
    st.image(result_image)
    for index, (emotion_name, score) in enumerate(emotions.items()):
       color = (255, 0,0) if score < 0.01 else (211, 211, 211)
       emotion_score = "{}: {}".format(emotion_name, "{:.2f}".format(score))
       
       st.subheader(emotion_score)
       
 
       
    
    
    
    cam.release()
