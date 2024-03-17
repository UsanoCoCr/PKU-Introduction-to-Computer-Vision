import numpy as np
from utils import draw_save_plane_with_points, normalize


if __name__ == "__main__":


    np.random.seed(0)
    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    # to simplify this problem, we provide the number of inliers and outliers here

    noise_points = np.loadtxt("HM1_ransac_points.txt")

    #RANSAC
    # Please formulate the palnace function as:  A*x+B*y+C*z+D=0 

    sample_time = 1000 # the minimal time that can guarantee the probability of at least one hypothesis does not contain any outliers is larger than 99.9%
    distance_threshold = 0.05

    # sample points group
    random_sample = np.random.choice(noise_points.shape[0], (sample_time, 3))
    random_sample = noise_points[random_sample] # (sample_time, 3, 3)
    # estimate the plane with sampled points group
    p1 = random_sample[:, 0]
    p2 = random_sample[:, 1]
    p3 = random_sample[:, 2]
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    A = normal[:, 0]
    B = normal[:, 1]
    C = normal[:, 2]
    D = -np.sum(normal * p1, axis=1)
    pf = np.array([A, B, C, D]).T # (sample_time, 4)

    #evaluate inliers (with point-to-plance distance < distance_threshold)
    inliers = np.zeros((sample_time, noise_points.shape[0]))
    inliers = np.abs(np.dot(pf, np.concatenate([noise_points.T, np.ones((1, noise_points.shape[0]))], axis=0))) < distance_threshold
    inliers = np.sum(inliers, axis=1)
    # print(inliers)

    # minimize the sum of squared perpendicular distances of all inliers with least-squared method 
    max_inliers = np.argmax(inliers)
    pf = pf[max_inliers]

    # draw the estimated plane with points and save the results 
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0  
    pf = normalize(pf)
    draw_save_plane_with_points(pf, noise_points,"result/HM1_RANSAC_fig.png") 
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)

