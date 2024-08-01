import cv2 as cv
import matplotlib.pyplot as plt

# 方法1.显示图片
def show_image(image, title, pos):
    # BGR 变为 RGB
    image_RGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    plt.subplot(2, 2, pos)
    plt.title(title)
    plt.imshow(image_RGB)
    plt.axis("off")
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
    # 1.读取图片
    img = cv.imread('lena.png')
    if img is None:
        print("Error: Image not found.")
        return
    # 2.转灰度
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 3.通过OpenCV自带的方法cv.CascadeClassifier()加载级联分类器
    """
    Haar cascade它提供了四个级联分类器(针对人脸的正面)
    (1)haarcascade_frontalface_alt.xml (FA1):22 stages and 20 x 20 haar features
    (2)haarcascade_frontalface_alt2.xml (FA2):20 stages and 20 x 20 haar features
    (3)haarcascade_frontalface_alt_tree.xml (FAT):47 stages and 20 x 20 haar features
    (4)haarcascade_frontalface_default.xml (FD):25 stages and 24 x 24 haar features
    """
    face_alt2 = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    # 4.对人脸进行检测
    face_alt2_detect = face_alt2.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # 5.绘制图片中的人脸
    face_alt2_result = plot_rectangle(img.copy(), face_alt2_detect)
    # 6.创建画布
    plt.figure(figsize=(9, 6))
    plt.suptitle('Face detection with Haar Cascade', fontsize=14, fontweight='bold')  # fontsize:字体大小，fontweight:字体粗细
    # 7.显示整个检测效果
    show_image(face_alt2_result, 'face_alt2', 1)
    plt.show()
if __name__ == "__main__":
    main()
