# --------------------------------------------------
# --------------> Resim Okuma-Yazma <---------------
import cv2


img = cv2.imread("./assets/where is ai used.jpg",0)
cv2.imshow("Resim Okuma", img)

k = cv2.waitKey(0)
if k == 27:
    print("ESC tuşuna basıldı.")
elif k == ord("q"):
    print("q tuşuna basıldı, resim kayıt edildi.")
    cv2.imwrite("grey_ai_image.jpg", img)

# cv2.destroyWindow("Resim Okuma")
cv2.destroyAllWindows()
# --------------------------------------------------


# --------------------------------------------------
# --------> Resim Okuma, PLT İle Gösterme <---------
import cv2
from matplotlib import pyplot as plt


img = cv2.imread("./assets/where is ai used.jpg")

plt.imshow(img)
plt.show()
# --------------------------------------------------


# --------------------------------------------------
# --------------> Pencere Oluşturma <---------------
import cv2


img = cv2.imread("./assets/GulBahcesi.jpg")

# cv2.namedWindow("Pencere Oluşturma",cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Pencere Oluşturma",cv2.WINDOW_NORMAL)
cv2.imshow("Pencere Oluşturma", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
# --------------------------------------------------


# --------------------------------------------------
# ---------------> Matris Oluşturma <---------------
import cv2
import numpy as np


matris_zeros = np.zeros([300,300])
matris_ones = np.ones([300,300])

cv2.namedWindow("Sifir Penceresi",cv2.WINDOW_NORMAL)
cv2.namedWindow("Bir Penceresi",cv2.WINDOW_NORMAL)
cv2.imshow("Sifir Penceresi", matris_zeros)
cv2.imshow("Bir Penceresi", matris_ones)

cv2.waitKey(0)
cv2.destroyAllWindows()
# --------------------------------------------------


# --------------------------------------------------
# ------------> Kameradan Görüntü Alma <------------
import cv2


camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Kamera tanınmadı.")
    exit()

while True:
    ret, frame = camera.read()

    if not ret:
        print("Kameradan goruntu okunamıyor!")
        break

    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    cv2.imshow("Kamera Goruntusu", frame)

    if cv2.waitKey(1) & 0xFF ==ord("q"):
        print("Goruntu sonlandırıldı.")
        break

camera.release()
cv2.destroyAllWindows()
# --------------------------------------------------


# --------------------------------------------------
# --------------> Kameradan Ayarlama <--------------
import cv2


camera = cv2.VideoCapture(1)

print(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
print(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

camera.set(cv2.CAP_PROP_FRAME_WIDTH,560)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,400)

print(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
print(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(camera.get(1))
print(camera.get(2))
print(camera.get(3))
print(camera.get(4))
print(camera.get(5))
print(camera.get(6))
print(camera.get(7))
print(camera.get(8))
print(camera.get(9))
print(camera.get(10))

camera.release()
# --------------------------------------------------


# --------------------------------------------------
# -----------------> Video Okuma <------------------
import cv2


camera = cv2.VideoCapture("./assets/Video Okuma.mp4")

while camera.isOpened():
    ret, frame = camera.read()

    if not ret:
        print("Video okunamıyor!")
        break

    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    cv2.imshow("Kamera Goruntusu", frame)

    if cv2.waitKey(1) & 0xFF ==ord("q"):
        print("Video kapatildi.")
        break

camera.release()
cv2.destroyAllWindows()
# --------------------------------------------------


# --------------------------------------------------
# -----------------> Video Yazma <------------------
import cv2


camera = cv2.VideoCapture(1)

fourrc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter("Read_image.avi", fourrc, 4.0, (640,480))

while camera.isOpened():
    ret, frame = camera.read()

    if not ret:
        print("Kameradan goruntu alınamıyor!")
        break

    out.write(frame)

    cv2.imshow("Kamera Goruntusu", frame)

    if cv2.waitKey(1) & 0xFF ==ord("q"):
        print("Kayit sonlandirildi..")
        break

camera.release()
out.release()
cv2.destroyAllWindows()
# --------------------------------------------------


# --------------------------------------------------
# ----------> Geometrik Şekil Oluşturma <-----------
import cv2
import numpy as np


img = np.zeros((512,512,3),np.uint8)

# cv2.line(img, (0,0), (511,511), (255,0,0),5)
# cv2.line(img, (50,400), (400,50), (0,255,0),5)

# cv2.rectangle(img, (50,50), (300,300), (0,0,255), 5)
# cv2.rectangle(img, (310,310), (511,511), (0,255,0), -1)

# cv2.circle(img, (255,255), 100, (100,100,100), 5)
# cv2.circle(img, (100,100), 60, (50,255,100), -1)

# cv2.ellipse(img, (256,256), (100,50), 0, 0, 360, (255,100,0), 5)
# cv2.ellipse(img, (256,100), (100,50), 0, 0, 360, (255,100,0), -1)

# plt = np.array([[20,30],[100,120],[255,255],[10,400]],np.int32)
# plt2= plt.reshape(-1,1,2)
# # cv2.polylines(img,[plt],False,(255,255,255),3)
# cv2.polylines(img,[plt],True,(255,255,255),3)

font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, "OpenCV", (10,400), font, 4, (0,155,255), 2, cv2.LINE_AA)

cv2.imshow("Geometrik Sekiller", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
# --------------------------------------------------


# --------------------------------------------------
# --> CV2 Kütüphanesine Ait EVENT Fonksiyonları <---
import cv2


for i in dir(cv2):
    if 'EVENT' in i:
        print(i)
# --------------------------------------------------


# --------------------------------------------------
# ----------> Fare Olayları (Çift Click) <----------
import cv2
import numpy as np


def draw(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x,y), 50,(50,150,250),-1)

img = np.ones((512,512,3),np.uint8)

cv2.namedWindow("Paint")
cv2.setMouseCallback("Paint", draw)

while True:
    cv2.imshow("Paint", img)

    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break

cv2.destroyAllWindows()
# --------------------------------------------------


# --------------------------------------------------
# ---------> Fare Olayları (Çizim Yapmak) <---------
import cv2
import numpy as np

isDrawing = False
isCircleMode = True
xi,yi = -1,-1

def draw(event, x, y, flags, param):
    global isDrawing, isCircleMode, xi, yi

    if event == cv2.EVENT_LBUTTONDOWN:
        xi,yi = x,y
        isDrawing=True

    elif event == cv2.EVENT_MOUSEMOVE:
        if isDrawing:
            if isCircleMode:
                cv2.circle(img, (x,y), 4, (100,50,0),-1)
            else:
                cv2.rectangle(img, (xi,yi), (x,y), (0,0,255),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        isDrawing = False

img = np.ones((512,512,3),np.uint8)

cv2.namedWindow("Paint")
cv2.setMouseCallback("Paint", draw)

while True:
    cv2.imshow("Paint", img)

    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break
    if cv2.waitKey(1) & 0xFF ==ord("m"):
        isCircleMode = not isCircleMode

cv2.destroyAllWindows()
# --------------------------------------------------


# --------------------------------------------------
# --------------> Trackbar Kullanımı <--------------
import cv2
import numpy as np


def nothing(x):
    pass

img = np.zeros((512,512,3),np.uint8)

cv2.namedWindow("Pencere")

cv2.createTrackbar("R", "Pencere", 0, 255, nothing)
cv2.createTrackbar("G", "Pencere", 0, 255, nothing)
cv2.createTrackbar("B", "Pencere", 0, 255, nothing)

cv2.createTrackbar("ON/OFF", "Pencere", 0, 1, nothing)

while True:
    cv2.imshow("Pencere", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    r = cv2.getTrackbarPos("R", "Pencere")
    g = cv2.getTrackbarPos("G", "Pencere")
    b = cv2.getTrackbarPos("B", "Pencere")

    switch = cv2.getTrackbarPos("ON/OFF", "Pencere")

    if switch:
        img[:] = [b,g,r]
    else:
        img[:] = 0

cv2.destroyAllWindows()
# --------------------------------------------------


# --------------------------------------------------
# -----------------> Resim Kırpma <-----------------
import cv2
import matplotlib.pyplot as plt


img = cv2.imread("./assets/where is ai used.jpg")

crop = img[175:410,375:650]

img[20:255,725:1000] = crop

plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(crop)
plt.show()
# --------------------------------------------------


# --------------------------------------------------
# ------------> İki Resmi Birleştirme <-------------
import cv2
import numpy as np


img1 = cv2.imread("./assets/where is ai used.jpg")
img2 = cv2.imread("./assets/GulBahcesi.jpg")

nested_img = cv2.addWeighted(img1[:500,:500],0.4,img2[:500,:500],0.6,0)

cv2.imshow("Birlesmis Resimler", nested_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
# --------------------------------------------------


# --------------------------------------------------
# ----------> Resim Üstüne Resim Koyma <------------
import cv2
import numpy as np


img1 = cv2.imread("./assets/where is ai used.jpg")
img2 = cv2.imread("./assets/openCV.png")

x, y, z = img2.shape
roi = img1[10:(x+10), 10:(y+10)]

img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2_gray,250,5,cv2.THRESH_TOZERO)

img2_bg = cv2.bitwise_and(roi,roi, mask=mask)

img2_fg = img2.copy()
img2_fg[np.where((img2_fg==[255,255,255]).all(axis=2))] = [0,0,0]

nested_img = cv2.add(img2_bg,img2_fg)

img1[10:(x+10), 10:(y+10)] = nested_img

cv2.namedWindow("Resim Ustune Resim Koyma",cv2.WINDOW_NORMAL)
cv2.imshow("Resim Ustune Resim Koyma", img1)

cv2.waitKey(0)
cv2.destroyAllWindows()
# --------------------------------------------------


# --------------------------------------------------
# -> Renk Uzayı ve Dönüşümü, Renkli Nesne Tespiti <-
import cv2
import numpy as np


camera = cv2.VideoCapture(0)

def nothing(x):
    pass

cv2.namedWindow("frame")
cv2.createTrackbar("H1","frame",0,359,nothing) # Renk uzayı 360 derecedir.
cv2.createTrackbar("H2","frame",0,359,nothing)
cv2.createTrackbar("S1","frame",0,255,nothing)
cv2.createTrackbar("S2","frame",0,255,nothing)
cv2.createTrackbar("V1","frame",0,255,nothing)
cv2.createTrackbar("V2","frame",0,255,nothing)

while camera.isOpened():
    _, frame =camera.read()

    hsv= cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    H1 = int(cv2.getTrackbarPos("H1","frame")/2) # OpenCV'de renk uzayı 180 derecedir.
    H2 = int(cv2.getTrackbarPos("H2","frame")/2)
    S1 = cv2.getTrackbarPos("S1","frame")
    S2 = cv2.getTrackbarPos("S2","frame")
    V1 = cv2.getTrackbarPos("V1","frame")
    V2 = cv2.getTrackbarPos("V2","frame")

    lower =np.array([H1,S1,V1])
    upper =np.array([H2,S2,V2])

    mask = cv2.inRange(hsv,lower,upper)

    res = cv2.bitwise_and(frame,frame,mask=mask)

    cv2.imshow("frame",frame)
    cv2.imshow("hsv",mask)
    cv2.imshow("res",res)

    if cv2.waitKey(5)==ord("q"):
        break

cv2.destroyAllWindows()
# --------------------------------------------------


# --------------------------------------------------
# ------------> Yeniden Boyutlandırma <-------------
import cv2
import numpy as np


img = cv2.imread("./assets/where is ai used.jpg")

res = cv2.resize(img, (300, 300))
res = cv2.resize(img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)

cv2.imshow("IMAGE", img)
cv2.imshow("RES", res)

cv2.waitKey(0)
cv2.destroyAllWindows()
# --------------------------------------------------


# --------------------------------------------------
# ----------------> Yer Değiştirme <----------------
import cv2
import numpy as np


img = cv2.imread("./assets/where is ai used.jpg")

rows, cols = img.shape[:2]

translation_matrix = np.float32([[1, 0, 25], [0, 1, 25]])

img_translation = cv2.warpAffine(img, translation_matrix, (cols+50, rows+50))

cv2.imshow("IMAGE", img)
cv2.imshow("IMAGE TRANSLATION", img_translation)

cv2.waitKey(0)
cv2.destroyAllWindows()
# --------------------------------------------------


# --------------------------------------------------
# ----------------> Resim Döndürme <----------------
import cv2
import numpy as np


img = cv2.imread("./assets/where is ai used.jpg")

rows, cols = img.shape[:2]

rotation_matrix = cv2.getRotationMatrix2D((cols/2,rows/2),17,0.7)

img_rotation = cv2.warpAffine(img, rotation_matrix, (cols, rows))

cv2.imshow("IMAGE", img)
cv2.imshow("IMAGE ROTATION", img_rotation)

cv2.waitKey(0)
cv2.destroyAllWindows()
# --------------------------------------------------


# --------------------------------------------------
# --------> Resmi Ölçeklendirme - Affine <----------
import cv2
import numpy as np


img = cv2.imread("./assets/where is ai used.jpg")

rows, cols = img.shape[:2]

src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])

dst_ponits = np.float32(
    [[0, 0], [int(0.6*(cols-1)), 0], [int(0.4*(cols-1)), rows-1]])

affline_matrix = cv2.getAffineTransform(src_points,dst_ponits)

img_output = cv2.warpAffine(img,affline_matrix,(cols,rows))

cv2.imshow("IMAGE", img)
cv2.imshow("IMAGE OUTPUT", img_output)

cv2.waitKey(0)
cv2.destroyAllWindows()
# --------------------------------------------------


# --------------------------------------------------
# ------> Resmi Ölçeklendirme - Perspective <-------
import cv2
import numpy as np


img = cv2.imread("./assets/where is ai used.jpg")

rows, cols = img.shape[:2]

src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])

dst_ponits = np.float32([[0, 0], [cols-1, 0], [int(0.33*(cols-1)), rows-1], [int(0.66*(cols-1)), rows-1]])

projective_matrix = cv2.getPerspectiveTransform(src_points, dst_ponits)

img_output = cv2.warpPerspective(img, projective_matrix, (cols, rows))

cv2.imshow("IMAGE", img)
cv2.imshow("IMAGE OUTPUT", img_output)

cv2.waitKey(0)
cv2.destroyAllWindows()
# --------------------------------------------------


# --------------------------------------------------
# -> Resmi Ölçeklendirme - Perspective-MouseClick <-
import cv2
import numpy as np


img = cv2.imread("./assets/where is ai used.jpg")

rows, cols = img.shape[:2]

click_count = 0
corners_list = []

dst_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])

cv2.namedWindow("IMAGE", cv2.WINDOW_NORMAL)
cv2.namedWindow("IMAGE OUTPUT", cv2.WINDOW_NORMAL)


def draw(event, x, y, flags, param):
    global click_count, corners_list

    if click_count < 4:
        if event == cv2.EVENT_LBUTTONDBLCLK:
            click_count += 1
            corners_list.append((x, y))
    else:
        src_points = np.float32([
            [corners_list[0][0], corners_list[0][1]],
            [corners_list[1][0], corners_list[1][1]],
            [corners_list[2][0], corners_list[2][1]],
            [corners_list[3][0], corners_list[3][1]]])

        click_count = 0
        corners_list = []

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        img_output = cv2.warpPerspective(img, matrix, (cols, rows))
        cv2.imshow("IMAGE OUTPUT", img_output)


cv2.setMouseCallback("IMAGE", draw)

while True:
    cv2.imshow("IMAGE", img)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
# --------------------------------------------------
