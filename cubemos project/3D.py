import matplotlib.pyplot as plt
import pickle
import numpy as np
import cv2

keypoint_ids = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17)
]


def get_valid_keypoints(keypoint_ids, skeleton, list_point, confs, confidence_threshold):
    keypoints = [
        (tuple(map(int, skeleton[i])), tuple(map(int, skeleton[v])))
        for (i, v) in keypoint_ids
        if confs[i] >= confidence_threshold
           and confs[v] >= confidence_threshold
    ]

    return keypoints


def load_var(load_path):
    file = open(load_path, 'rb')
    variable = pickle.load(file)
    file.close()
    return variable


def save_var(save_path, variable):
    file = open(save_path, 'wb')
    pickle.dump(variable, file)
    print("variable saved.")
    file.close()


frame_list = load_var("point_cordinate.pckl")
frame_idx = 102
my_frame = frame_list[frame_idx]
fig = plt.figure(figsize=(1, 1))
ax = plt.axes(projection='3d')
img = np.zeros(shape=[512, 512, 3], dtype=np.uint8)

X = []
Y = []
Z = []
X1 = []
Y1 = []
Z1 = []
points, skeletons = frame_list[frame_idx]
list_point = list(points.values())
confs = [x[2] for x in skeletons]
skeleton = [x[:2] for x in skeletons]

keypoints = get_valid_keypoints(keypoint_ids, skeleton, list_point, confs, 0.2)

#
# list_point = list(points.values())
img = np.zeros(shape=[512, 512, 3], dtype=np.uint8)

for keypoint in keypoints:
    X = []
    Y = []
    Z = []
    key1 = keypoint[0]
    key2 = keypoint[1]
    (x, y, z) = points[key1]
    (x1, y1, z1) = points[key2]
    X.append(x)
    X.append(x1)
    Y.append(y)
    Y.append(y1)
    Z.append(z)
    Z.append(z1)
    ax.plot3D(X, Y, Z)

for i in range(len(list_point)):
    (x, y, z) = list_point[i]
    X1.append(x)
    Y1.append(y)
    Z1.append(z)

ax.scatter(X1, Y1, Z1)

for angle in range(0, 360):
    ax.view_init(angle, 30)
    plt.draw()

ax.set_xlabel('x_camera')
ax.set_ylabel('y_camera')
ax.set_zlabel('z_camera')

plt.show()
