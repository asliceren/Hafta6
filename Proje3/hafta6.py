import os
import cv2
import numpy as np

dosya_adi = "ornek.jpg"
tam_yol = os.path.join(os.path.dirname(os.path.abspath(__file__)), dosya_adi)

img = cv2.imread(tam_yol, cv2.IMREAD_COLOR)
if img is None:
    img = cv2.imread(dosya_adi, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Goruntu bulunamadi: {tam_yol}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, threshold1=50, threshold2=150)
median_edges = cv2.medianBlur(edges, ksize=3)

circles = cv2.HoughCircles(
    median_edges,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=30,
    param1=100,
    param2=30,
    minRadius=10,
    maxRadius=150
)

output_img = img.copy()

if circles is not None:
    circles_uint16 = np.uint16(np.around(circles))
    for cx, cy, r in circles_uint16[0, :]:
        cv2.circle(output_img, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
        cv2.circle(output_img, (int(cx), int(cy)), 2, (255, 0, 255), 3)

pencere_isimleri = [
    "1. Canny Kenarlar",
    "2. Median Filtreli Kenarlar",
    "3. HoughCircles Sonuc"
]
goruntuler = [
    edges,
    median_edges,
    output_img
]

for ad, goruntu in zip(pencere_isimleri, goruntuler):
    cv2.namedWindow(ad, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ad, 650, 650)
    cv2.imshow(ad, goruntu)

cv2.waitKey(0)
cv2.destroyAllWindows()
