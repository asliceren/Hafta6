import cv2
import numpy as np

img = cv2.imread("ornek.jpg", cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError("Goruntu bulunamadi! Dosya yolunu kontrol edin.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, threshold1=50, threshold2=150)
median_edges = cv2.medianBlur(edges, ksize=3)

img_hough_standard = img.copy()
img_hough_prob = img.copy()

lines_std = cv2.HoughLines(median_edges, rho=1, theta=np.pi / 180, threshold=120)

if lines_std is not None:
    for line in lines_std:
        rho, theta = np.asarray(line).reshape(-1)[:2]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(img_hough_standard, pt1, pt2, (0, 0, 255), 2)

lines_prob = cv2.HoughLinesP(
    median_edges,
    rho=1,
    theta=np.pi / 180,
    threshold=80,
    minLineLength=50,
    maxLineGap=10
)

if lines_prob is not None:
    for line in lines_prob:
        x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]
        cv2.line(img_hough_prob, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

cv2.imshow("1. Canny Kenarlar", edges)
cv2.imshow("2. Median Filtreli Kenarlar", median_edges)
cv2.imshow("3. cv2.HoughLines (Kirmizi)", img_hough_standard)
cv2.imshow("4. cv2.HoughLinesP (Mavi)", img_hough_prob)

cv2.waitKey(0)
cv2.destroyAllWindows()