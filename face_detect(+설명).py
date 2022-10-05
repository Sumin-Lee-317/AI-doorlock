# MIT License
# Copyright (c) 2019-2022 JetsonHacks
# See LICENSE for OpenCV license and additional information

# https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_face_detection.html
# On the Jetson Nano, OpenCV comes preinstalled
# Data files are in /usr/sharc/OpenCV

import cv2

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
## gstreamer_pipeline: CSI 카메라에서 캡처할 GStreamer 파이프라인을 반환합니다.
## Defaults to 1920x1080 @ 30fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
## flip_method(가장 일반적인 값: 0 및 2)를 설정하여 이미지를 뒤집습니다
# display_width and display_height determine the size of the window on the screen
## display_width 및 display_height는 화면의 창 크기를 결정합니다.
# Notice that we drop frames if we fall outside the processing time in the appsink element
# 앱 싱크 요소에서 처리 시간을 벗어나면 프레임이 손실됩니다.


## gstreamer_pipeline
# CSI 카메라에서 캡처할 GStreamer 파이프라인을 반환합니다.

def gstreamer_pipeline(
    capture_width=1920, # capture 캡처할 사진의 너비와 높이 지정
    capture_height=1080,
    display_width=960, # display 캡처한 사진을 출력할 크기 지정
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def face_detect():
    window_title = "Face Detect" # 창의 이름 지정
    # 얼굴과 눈을 감지하기 위해 Cascade 분류기를 사용함 (얼굴 검출 용도로 많이 쓰임)
    
    # 사전학습된 정면 얼굴 검출용 XML 파일을 CascadeClassifier() 함수를 사용해 얼굴 객체로 생성해서 face_cascade 변수에 저장
    face_cascade = cv2.CascadeClassifier(
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    )
    
    # 사전학습된 눈 검출용 XML 파일을 CascadeClassifier() 함수를 사용해 눈 객체로 생성해서 eye_cascade 변수에 저장
    eye_cascade = cv2.CascadeClassifier(
        "/usr/share/opencv4/haarcascades/haarcascade_eye.xml"
    )
    
    # 파란색 사각형 = 얼굴 검출 결과 / 녹색 사각형 = 눈 검출 결과
    
    video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER) # 위에서 설정한 카메라 열기 
    if video_capture.isOpened(): # 만약 (카메라)비디오 캡처가 준비되었으면, 
        try: # try 부분 우선 수행
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE) # 아까 설정한 window_title로 기본 크기의 창 생성
            while True: # 항상 실행
                ret, frame = video_capture.read() # 동영상 파일을 읽어오고 영상의 각 프레임에서 이미지를 추출하여 frame에 저장
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 캡처본을 불러와서 cvtColor() 함수를 사용해 BGR 형식을 GRAY 형식으로 변환
                faces = face_cascade.detectMultiScale(gray, 1.3, 5) # face_cascade 객체: scaleFactor 를 1.3, minNeighbors 를 5로 설정

                # 각각의 행마다 (x,y,w,h) 받아와서 얼굴에 사각형을 그리는 코드
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    roi_gray = gray[y : y + h, x : x + w]
                    roi_color = frame[y : y + h, x : x + w]
                    eyes = eye_cascade.detectMultiScale(roi_gray) # detectMultiScale() 함수를 사용해 객체를 탐지
                    # 각각의 행마다 (x,y,w,h) 받아와서 눈에 녹색 사각형을 그리는 코드
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(
                            roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
                        )
                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                else:
                    break
                keyCode = cv2.waitKey(10) & 0xFF 
                # ESC 키나 q 키를 누르면 프로그램을 중지하는 코드
                if keyCode == 27 or keyCode == ord('q'): # 키 입력을 받으면 키 값을 key로 저장 -> esc == 27(아스키코드)
                    break # while문 빠져나가기
        finally: # 예외의 발생 여부와는 상관없이 항상 실행되는 부분
            video_capture.release() # 영상 파일(카메라) 사용을 종료
            cv2.destroyAllWindows() # 모든 창 닫기
    else:
        print("Unable to open camera") # 오류 발생시 카메라를 열 수 없다는 메세지 출력


if __name__ == "__main__":
    face_detect()
