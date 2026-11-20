import os
import cv2
import numpy as np

dosya_adi = "ornek.jpg"
tam_yol = os.path.join(os.path.dirname(os.path.abspath(__file__)), dosya_adi)

if not os.path.exists(tam_yol):
    raise FileNotFoundError(f"Dosya bulunamadi: {tam_yol}")

veri = np.fromfile(tam_yol, dtype=np.uint8)
img = cv2.imdecode(veri, cv2.IMREAD_COLOR)

if img is None:
    raise ValueError(f"Dosya bozuk veya desteklenmeyen formatta: {tam_yol}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

height, width = edges.shape

min_r = 15
max_r = 80
r_step = 2
radii = np.arange(min_r, max_r + 1, r_step)
num_radii = len(radii)

thetas = np.deg2rad(np.arange(0, 360, 5))
cos_table = np.cos(thetas)
sin_table = np.sin(thetas)

accumulator = np.zeros((height, width, num_radii), dtype=np.int32)

edge_y, edge_x = np.nonzero(edges)
print("Toplam kenar pikseli sayisi:", len(edge_x))

for x, y in zip(edge_x, edge_y):
    for r_idx, r in enumerate(radii):
        a_vals = np.round(x - r * cos_table).astype(np.int32)
        b_vals = np.round(y - r * sin_table).astype(np.int32)
        for a, b in zip(a_vals, b_vals):
            if 0 <= b < height and 0 <= a < width:
                accumulator[b, a, r_idx] += 1

acc_max = int(accumulator.max())
print("Maksimum oy sayisi:", acc_max)

threshold = int(0.55 * acc_max) if acc_max > 0 else 20
print("Kullanilan esik degeri:", threshold)

result_img = img.copy()
b_indices, a_indices, r_indices = np.where(accumulator >= threshold)
print("Esigi gecen aday sayisi:", len(a_indices))

detected_circles = []
min_center_dist = 20

for b, a, r_idx in zip(b_indices, a_indices, r_indices):
    cx, cy, r = int(a), int(b), int(radii[r_idx])
    too_close = False
    for pcx, pcy, pr in detected_circles:
        dist = np.hypot(cx - pcx, cy - pcy)
        if dist < min_center_dist and abs(r - pr) < 15:
            too_close = True
            break
    if not too_close:
        detected_circles.append((cx, cy, r))
        cv2.circle(result_img, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(result_img, (cx, cy), 3, (0, 0, 255), -1)

pencere_isimleri = [
    "Kenarlar",
    "Bulunan Cemberler"
]
goruntuler = [
    edges,
    result_img
]

for ad, goruntu in zip(pencere_isimleri, goruntuler):
    cv2.namedWindow(ad, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ad, 650, 650)
    cv2.imshow(ad, goruntu)

cv2.waitKey(0)
cv2.destroyAllWindows()
