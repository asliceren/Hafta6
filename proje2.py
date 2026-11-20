import math
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
edges = cv2.Canny(gray, 50, 150)

height, width = edges.shape
max_rho = int(math.ceil(math.hypot(width, height)))
num_rhos = 2 * max_rho + 1

thetas_deg = np.arange(-90, 90, 1)
thetas = np.deg2rad(thetas_deg)
num_thetas = len(thetas)

cos_table = np.cos(thetas)
sin_table = np.sin(thetas)

accumulator = np.zeros((num_rhos, num_thetas), dtype=np.int32)

edge_y, edge_x = np.nonzero(edges)
print("Toplam kenar pikseli sayisi:", len(edge_x))

for x, y in zip(edge_x, edge_y):
    for t_idx in range(num_thetas):
        rho_val = x * cos_table[t_idx] + y * sin_table[t_idx]
        rho_idx = int(round(rho_val)) + max_rho
        if 0 <= rho_idx < num_rhos:
            accumulator[rho_idx, t_idx] += 1

acc_max = int(accumulator.max())
print("Maksimum oy sayisi:", acc_max)

if acc_max > 0:
    acc_norm = ((accumulator.astype(np.float32) / acc_max) * 255.0).astype(np.uint8)
else:
    acc_norm = np.zeros_like(accumulator, dtype=np.uint8)

acc_visual = cv2.resize(
    acc_norm,
    (num_thetas * 4, min(num_rhos, 900)),
    interpolation=cv2.INTER_NEAREST
)
cv2.imwrite("oylama_matrisi.png", acc_visual)

threshold = int(0.45 * acc_max) if acc_max > 0 else 50
print("Kullanilan esik degeri:", threshold)

result_img = img.copy()
rho_indices, theta_indices = np.where(accumulator >= threshold)
print("Esigi gecen hucre sayisi:", len(rho_indices))

for r_idx, t_idx in zip(rho_indices, theta_indices):
    rho = r_idx - max_rho
    a = cos_table[t_idx]
    b = sin_table[t_idx]
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(round(x0 + 2000 * (-b))), int(round(y0 + 2000 * a)))
    pt2 = (int(round(x0 - 2000 * (-b))), int(round(y0 - 2000 * a)))
    cv2.line(result_img, pt1, pt2, (0, 0, 255), 2)

pencere_isimleri = [
    "Kenarlar",
    "Oylama Matrisi",
    "Bulunan Dogrular"
]
goruntuler = [
    edges,
    acc_visual,
    result_img
]

for ad, goruntu in zip(pencere_isimleri, goruntuler):
    cv2.namedWindow(ad, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ad, 650, 650)
    cv2.imshow(ad, goruntu)

cv2.waitKey(0)
cv2.destroyAllWindows()
