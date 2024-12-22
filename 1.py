import rospy
import cv2
import os
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from datetime import datetime

if not os.path.exists('screenshots'):
    os.makedirs('screenshots')

face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("Failed to load cascade classifier.")
    exit()

recognized_faces = {}
face_id_counter = 0
specified_face_id = None  

bridge = CvBridge()

def save_face_info(face_id, face_roi):
    recognized_faces[face_id] = face_roi

def compare_faces(face1, face2):
    hist1 = cv2.calcHist([face1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([face2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def recognize_faces(faces, gray_frame, frame):
    global face_id_counter, specified_face_id
    processed_ids = set()  
    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y+h, x:x+w]
        face_found = False
      
        if specified_face_id is not None and specified_face_id not in processed_ids:
            for id, known_face in recognized_faces.items():
                resized_face_roi = cv2.resize(face_roi, (known_face.shape[1], known_face.shape[0]))
                if id == specified_face_id and compare_faces(resized_face_roi, known_face) > 0.4:
                    face_found = True
                    processed_ids.add(id)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) 
                    cv2.putText(frame, f'Specified ID: {id}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    break

        if not face_found:
            for id, known_face in recognized_faces.items():
                resized_face_roi = cv2.resize(face_roi, (known_face.shape[1], known_face.shape[0]))
                if compare_faces(resized_face_roi, known_face) > 0.75: 
                    face_found = True
                    processed_ids.add(id)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) 
                    cv2.putText(frame, f'ID: {id}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    break

        if not face_found and face_id_counter not in processed_ids:
            face_id_counter += 1
            save_face_info(face_id_counter, face_roi)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {face_id_counter}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

class ImageConverter:
    def __init__(self):
        # 初始化ROS节点
        self.image_pub = rospy.Publisher("cv_bridge_image", Image, queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)
        
        # 初始化窗口
        cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)

    def callback(self, data):
        # 使用cv_bridge将ROS的图像数据转换成OpenCV的图像格式
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        recognize_faces(faces, gray_frame, frame)

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y+h, x:x+w]
            face_found = False
            for id, known_face in recognized_faces.items():
                resized_face_roi = cv2.resize(face_roi, (known_face.shape[1], known_face.shape[0]))
                if compare_faces(resized_face_roi, known_face) > 0.75:
                    face_found = True
                    break
            if not face_found:  
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                face_filename = f'screenshots/face_{timestamp}.png'
                cv2.imwrite(face_filename, face_roi)
                print(f"Saved face to {face_filename}")

        cv2.imshow('Face Recognition', frame)
        cv2.waitKey(1) 

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        except CvBridgeError as e:
            print(e)

if __name__ == '__main__':
    try:
        rospy.init_node("face_recognition_node")
        rospy.loginfo("Starting face recognition node")
        ic = ImageConverter()

        while not rospy.is_shutdown():
            key = input("输入 's' 进行人脸搜索: ")
            if key == 's':
                face_id = input("输入人脸ID: ")
                if face_id.isdigit() and int(face_id) in recognized_faces:
                    specified_face_id = int(face_id)
                    print(f"已指定人脸ID {specified_face_id}。")
                else:
                    print(f"人脸ID {face_id} 未识别。")
            rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down face recognition node.")
        cv2.destroyAllWindows()
