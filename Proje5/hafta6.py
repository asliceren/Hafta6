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
    raise ValueError(f"Dosya okunamadi: {tam_yol}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

height, width = edges.shape

min_r = 15
max_r = 90
r_step = 3
radii = np.arange(min_r, max_r + 1, r_step)
num_radii = len(radii)

thetas = np.deg2rad(np.arange(0, 360, 5))
cos_table = np.cos(thetas)
sin_table = np.sin(thetas)

accumulator = np.zeros((height, width, num_radii), dtype=np.int32)
concentric_weight = 0.25

edge_y, edge_x = np.nonzero(edges)
print("Kenar pikseli sayisi:", len(edge_x))

for x, y in zip(edge_x, edge_y):
    for r_idx, r in enumerate(radii):
        a_vals = np.round(x - r * cos_table).astype(np.int32)
        b_vals = np.round(y - r * sin_table).astype(np.int32)
        for a, b in zip(a_vals, b_vals):
            if 0 <= b < height and 0 <= a < width:
                other_radii_votes = int(accumulator[b, a, :].sum() - accumulator[b, a, r_idx])
                bonus = int(concentric_weight * max(0, other_radii_votes))
                accumulator[b, a, r_idx] += 1 + bonus

acc_max = int(accumulator.max())
print("Maksimum oy degeri:", acc_max)

threshold = int(0.55 * acc_max) if acc_max > 0 else 20
print("Esik degeri:", threshold)

result_img = img.copy()
b_indices, a_indices, r_indices = np.where(accumulator >= threshold)
print("Esigi gecen cember sayisi:", len(a_indices))

for b, a, r_idx in zip(b_indices, a_indices, r_indices):
    cx = int(a)
    cy = int(b)
    r = int(radii[r_idx])
    cv2.circle(result_img, (cx, cy), r, (0, 255, 0), 2)
    cv2.circle(result_img, (cx, cy), 3, (0, 0, 255), -1)

cv2.namedWindow("Kenarlar", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Kenarlar", 600, 600)
cv2.imshow("Kenarlar", edges)

cv2.namedWindow("Es Merkezli Cemberler", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Es Merkezli Cemberler", 600, 600)
cv2.imshow("Es Merkezli Cemberler", result_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
