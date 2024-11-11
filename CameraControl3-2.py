#최종 수정본
import cv2
from ultralytics import YOLO
import numpy as np

from gtts import gTTS
import pygame
#-*- coding: utf-8 -*-

def Helmet_in_FOI(h, t) :
	print("헬멧 위치 : ", h, "FOI 영역 : ", t)
	return np.all(h >= t[:2]) and np.all(h <= t[2:])

# TTS 음성 출력========================
def text_speech(text):
    text = "헬멧을 착용해주세요."
    tts = gTTS(text = text, lang="ko")
    filename = "C:\\Users\\happy\\Desktop\\CameraControl\\헬멧을 착용해주세요.mp3"

    # tts.save(filename)

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        continue

# =====================================

model_path = 'C:\\Users\\happy\\runs\\best.pt'
WIDTH = 640
HEIGHT = 480

def main():
    
    print("Check Camera...", end=' ')
    wide_cam = cv2.VideoCapture(0) # 광각 카메라
    wide_cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)  # 광각 카메라의 프레임 너비 설정
    wide_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)  # 광각 카메라의 프레임 높이 설정

    while True:
            print('.', end='')  # 진행 중 점 출력
            ret, frame = wide_cam.read()
            ret, frame = wide_cam.read()

                
            if ret:
                cv2.imshow("Captured", frame)
                if (cv2.waitKey(10) == 27) :
                    break

            image = np.asarray(frame)
            found = model1.predict(image, show=True, conf=0.7, verbose=False) 
            

            for result in found:# 라이더 여러명, 헬멧 여러개인 것을 확인
	           # (H1, H2, R1, R2, H3) : R2-H2, H1과 H3는 무관한 헬멧 
                for box in result.boxes:
                        id = box.cls.cpu().numpy()
                        npp = box.xyxyn.cpu().numpy() * np.array([WIDTH, HEIGHT, WIDTH, HEIGHT])
                        
                        if id == 0: # 드라이버
                            driver_foi = npp
                            driver_foi[0][3] = npp[0][1] + int(npp[0][3] - npp[0][1]/2)
                            driver_foi = driver_foi.astype(int)


                if driver_foi is not None:
                    helmet_found = False
                    for r in result.boxes:
                        r_id = r.cls.cpu().numpy()
                        r_npp = r.xyxyn.cpu().numpy() * np.array([WIDTH, HEIGHT, WIDTH, HEIGHT])
                        r_npp = r_npp.astype(int)
                        
                        if r_id in [1, 2]:  # 헬멧
                            if Helmet_in_FOI(r_npp, driver_foi):
                                helmet_found = True
                                break        
                                
                                    
                            
                        # 헬멧을 찾지 못했으면 (helmet_found가 False일 경우)
                        if not (helmet_found):                                        
                            face = frame[driver_foi[0][1] : driver_foi[0][3], driver_foi[0][0] : driver_foi[0][2]]
                            # 상반신 부분을 2배 확대하여 표시 
                            face = cv2.resize(face, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

                            # TTS 음성 출력 위치======================
                            # f = open("헬멧을 착용해주세요..mp3", "r", encoding = 'utf-8')
                            # f.read()
                            text_speech("헬멧을 착용해주세요..mp3")
                            cv2.imshow('Face', face)
                            c = cv2.waitKey(1)
                            if (c == 27):  # ESC 키가 눌리면 루프 종료
                                break
                            elif (c != ord('s') and c != ord('S')):  # 's'나 'S' 키가 아니면 계속 루프 진행
                                continue  # 키 입력을 1ms 기다림

                            print("===>", time.time())  # 현재 시간을 출력

    # 프레임을 JPEG 형식으로 인코딩 (품질은 90으로 설정)
                            retval, frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            print(retval)  # 인코딩 성공 여부 출력
                                        
                             
    cv2.destroyAllWindows()
    wide_cam.release()
       
          
if __name__ == "__main__" :
    # 학습한 model 읽어 들임.
    print("motorcycle Detector model is loading...")    
    model1 = YOLO('C:\\Users\\happy\\runs\\best.pt')

    main()