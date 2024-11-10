#최종 수정본
import cv2
from ultralytics import YOLO
import numpy as np

def Helmet_in_FOI(h, t) :
	print("헬멧 위치 : ", h, "FOI 영역 : ", t)
	return (False)

#model2 = YOLO('C:\\Users\\happy\\runs\\best.pt')
WIDTH = 640
HEIGHT = 480

def main():  # 메인 함수 정의
    
    print("Check Camera...", end=' ')
    wide_cam = cv2.VideoCapture(0) # 광각 카메라
    wide_cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH, 640)  # 광각 카메라의 프레임 너비 설정
    wide_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT, 480)  # 광각 카메라의 프레임 높이 설정

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
                        npp = box.xyxyn.cpu().numpy() #Bounding box 좌표  # 좌표 변환
                        npp = npp * np.array([640, 480, 640, 480])
                        
                        if id == 0: # 드라이버
                            driver_foi = npp
                            
                            driver_foi[0][3] = npp[0][1] + int(npp[0][3] - npp[0][1]/2)
                            


                        # for r in found:
                            helmet_found = False
                            for r in result.boxes:
                                r_id = r.cls.cpu().numpy()
                                r_npp = r.xyxyn.cpu().numpy()
                                r_npp = r_npp * np.array([640, 480, 640, 480])
						                                 
                                if r_id in [1, 2]: #헬멧
                                    if Helmet_in_FOI(r_npp.astype(int), foi.astype(int)) == True :
                                        helmet_found = True
                                        break
		
	
                                # if driver_foi is not None:
                                # 헬멧이 드라이버의 바운딩 박스 안에 있는지를 확인
                                
                                    
                            
                                # 헬멧을 찾지 못했으면 (helmet_found가 False일 경우)
                        if not (helmet_found):
                                
                            foi = foi.astype(int)
                                        
                            face = frame[foi[0][1]0 : foi[0][3], foi[0][0], foi[0][2]]
                            # 상반신 부분을 2배 확대하여 표시 
                            face = cv2.resize(face, dsize=(0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
                        cv2.imshow('Face', face)
                        c = cv2.waitKey(1)  # 키 입력을 1ms 기다림
                                        
                             
    cv2.destroyAllWindows()
    wide_cam.release()
       
          
if __name__ == "__main__" :

# 학습한 model 읽어 들임.
    print("motorcycle Detector model is loading...")    
    model1 = YOLO('C:\\Users\\happy\\runs\\best.pt')

# main() 함수 호출 --> 프로그램 시작점   
    main()