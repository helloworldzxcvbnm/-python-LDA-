from model import model
from PIL import Image
import numpy as np



def classify(a, b, li):
    white = 0
    black = 0

    for i in li:
        for j in i:
            if j == 0:
                black += 1
            else:
                white += 1

    temp = black
    black = round(black / (black + white), 3)  # 计算黑色区域占比
    white = round(white / (temp + white), 3)  # 计算白色区域占比

    if (a * black + b) < white:  # 进行判断
        return False
    else:
        return True


def draw(x_y, pic):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    # 读取图片
    img = mpimg.imread(pic)

    # 显示图片
    plt.imshow(img)

    # 画点，例如在(x, y)处画一个红色的点
    plt.imshow(img, extent=[0, img.shape[1], img.shape[0], 0], cmap='gray')
    for x, y in x_y:
        plt.scatter(y, x, color='red', s=70)
    # 显示画好点的图片
    plt.show()


def get_li(pic):
    image = Image.open(pic)
    gray_image = image.convert("L")
    threshold = 100
    binary_image = gray_image.point(lambda x: 255 if x >= threshold else 0, mode="1")
    image_array = np.array(binary_image)
    image_array = image_array.astype(np.uint8)
    a = image_array.tolist()
    return list(a)


def run(pic):
    li = list(get_li(pic))  # 把图片转换为对于的二值 0 1 列表
    x_y = []  # 初始化图片中损失点的坐标集合
    a, b = model()  # 获得LDA模型的直线方程的 a, b (y=ax+b)

    pian_x = 0  # 设置分割图片时，起始x, y 的偏移量
    pian_y = 0

    temp_x = len(li) // 60  # x的长度为60
    temp_y = len(li[0]) // 70  # y的长度为70

    for i in range(temp_x):  # 循环遍历
        for j in range(temp_y):
            temp = [k[pian_y+70*j:pian_y+70*j+70] for k in li[pian_x+60*i:pian_x+60*i+60]]  # 通过列表推导式以及切片获取对应分割后图片的二值 0 1 列表
            if not classify(a, b, temp):  # 进行距离判别
                x_y.append([pian_x+i*60+30, pian_y+j*70+35])
    draw(x_y, pic)  # 传入损失的点的坐标集合和图片路径，通过draw函数绘制标出损失后的图像


for i in range(1, 6):
    pic = f"img/img_{i}.png"
    run(pic)

