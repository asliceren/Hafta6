import math
import time
import cv2
import numpy as np

width = 256
height = 256

img = np.zeros((height, width), dtype=np.uint8)
img[80, 20:236] = 255
img[20:236, 140] = 255
for i in range(30, 200):
    img[i, i] = 255

edge_y, edge_x = np.nonzero(img > 0)
num_edges = len(edge_x)
print("Kenar pikseli sayisi:", num_edges)

theta_min_deg = -90.0
theta_step_deg = 1.0
thetas_deg = np.arange(theta_min_deg, 90.0, theta_step_deg)
thetas = np.deg2rad(thetas_deg)
num_thetas = len(thetas)

cos_table = np.cos(thetas).astype(np.float32)
sin_table = np.sin(thetas).astype(np.float32)

max_rho = int(math.ceil(math.hypot(width, height)))
num_rhos = 2 * max_rho + 1
accumulator = np.zeros((num_rhos, num_thetas), dtype=np.int32)

t0 = time.perf_counter()

ex = edge_x.astype(np.float32)[:, None]
ey = edge_y.astype(np.float32)[:, None]

rho_vals = ex * cos_table[None, :] + ey * sin_table[None, :]
rho_indices = np.rint(rho_vals).astype(np.int32) + max_rho

valid = (rho_indices >= 0) & (rho_indices < num_rhos)
theta_indices = np.broadcast_to(np.arange(num_thetas, dtype=np.int32)[None, :], rho_indices.shape)

flat_votes = np.bincount(
    rho_indices[valid] * num_thetas + theta_indices[valid],
    minlength=num_rhos * num_thetas
)
accumulator = flat_votes.reshape((num_rhos, num_thetas)).astype(np.int32)

elapsed_ms = (time.perf_counter() - t0) * 1000.0
print(f"Hesaplama suresi: {elapsed_ms:.2f} ms")

max_votes = int(accumulator.max())
print("Maksimum oy:", max_votes)

threshold = int(0.75 * max_votes)
r_idx_list, t_idx_list = np.where(accumulator >= threshold)
print("Esigi gecen hucre sayisi:", len(r_idx_list))

for r_idx, t_idx in zip(r_idx_list, t_idx_list):
    rho = r_idx - max_rho
    theta_deg = thetas_deg[t_idx]
    votes = accumulator[r_idx, t_idx]
    print(f"Dogru -> rho: {rho}, theta: {theta_deg:.1f} deg, oy: {votes}")

color_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for r_idx, t_idx in zip(r_idx_list, t_idx_list):
    rho = r_idx - max_rho
    a = float(cos_table[t_idx])
    b = float(sin_table[t_idx])
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(round(x0 + 1000 * (-b))), int(round(y0 + 1000 * a)))
    pt2 = (int(round(x0 - 1000 * (-b))), int(round(y0 - 1000 * a)))
    cv2.line(color_out, pt1, pt2, (0, 0, 255), 1)

cv2.namedWindow("Bulunan Dogrular", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Bulunan Dogrular", 500, 500)
cv2.imshow("Bulunan Dogrular", color_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
