import numpy as np
from utils import draw_save_plane_with_points, normalize


if __name__ == "__main__":


    np.random.seed(0)
    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    # to simplify this problem, we provide the number of inliers and outliers here

    noise_points = np.loadtxt("HM1_ransac_points.txt")

    #RANSAC
    # Please formulate the palnace function as:  A*x+B*y+C*z+D=0 

    sample_time = 12 # the minimal time that can guarantee the probability of at least one hypothesis does not contain any outliers is larger than 99.9%
    distance_threshold = 0.05

    # sample points group
    # num_points = noise_points.shape[0]
    random_sample = np.random.choice(noise_points.shape[0], (sample_time, 3))
    random_sample = noise_points[random_sample] # (sample_time, 3, 3)
    # estimate the plane with sampled points group
    p1 = random_sample[:, 0]
    p2 = random_sample[:, 1]
    p3 = random_sample[:, 2]
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2) # (sample_time, 3)
    A = normal[:, 0]
    B = normal[:, 1]
    C = normal[:, 2]
    D = -np.sum(normal * p1, axis=1)
    pf = np.array([A, B, C, D]).T # (sample_time, 4)

    #evaluate inliers (with point-to-plance distance < distance_threshold)
    norms = np.linalg.norm(normal, axis=1) # (sample_time,)
    distance = np.abs(np.dot(noise_points, normal.T) + D) / norms # (point_num, sample_time)
    inliers = np.zeros_like(distance) # (point_num, sample_time)
    inliers[distance < distance_threshold] = 1
    inliers_count = np.sum(inliers, axis=0) # (sample_time,)
    max_inliers = np.argmax(inliers_count)
    # print(inliers_count[max_inliers])

    # minimize the sum of squared perpendicular distances of all inliers with least-squared method 
    sample = noise_points[inliers[:, max_inliers] == 1]
    x, y, z = sample[:, 0], sample[:, 1], sample[:, 2]
    mean_x, mean_y, mean_z = np.mean(x), np.mean(y), np.mean(z)
    x = x - mean_x
    y = y - mean_y
    z = z - mean_z
    A = np.array([x, y, z]).T
    U, S, Vh = np.linalg.svd(A)
    Vh = Vh.T
    h = Vh[:, -1]
    # print(h)
    d = -(h[0] * mean_x + h[1] * mean_y + h[2] * mean_z)
    pf = np.array([h[0], h[1], h[2], d])
    pf = normalize(pf)
    # print(pf)

    # draw the estimated plane with points and save the results 
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0  
    draw_save_plane_with_points(pf, noise_points,"result/HM1_RANSAC_fig.png") 
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)

