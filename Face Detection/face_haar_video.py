import cv2 as cv

# 方法2.绘制图片中检测到的人脸
def plot_rectangle(image, faces):
    # 得到的faces人脸数据有四个值：坐标（x, y），宽高width,height
    for (x, y, w, h) in faces:
        #                   框起始点     框对角点   颜色         粗细
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
        # 在框上添加文字       文字            文字的左下角坐标  字体类型                字体大小    字体颜色  字体粗细
        cv.putText(image, 'Straight Face', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return image

# 主函数
def main():
    # 1.读取摄像头
    capture = cv.VideoCapture(0)
    # 2.判断摄像头是否工作
    if capture.isOpened() is False:
        print('Camera Error !')
    while True:
        # 获取每一帧
        ret, frame = capture.read()
        if ret is True:
            img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # 3.通过OpenCV自带的方法cv.CascadeClassifier()加载级联分类器
            face_alt2 = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
            # 4.对人脸进行检测(只能检测正脸)
            face_alt2_detect = face_alt2.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            # 5.绘制图片中的人脸
            face_alt2_result = plot_rectangle(frame.copy(), face_alt2_detect)
            # 6.在视频中显示
            cv.imshow('Face detection', face_alt2_result)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
    capture.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
