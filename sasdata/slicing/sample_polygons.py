import numpy as np


def wedge(q0, q1, theta0, theta1, clockwise=False, n_points_per_degree=2):

    # Traverse a rectangle in curvilinear coordinates (q0, theta0), (q0, theta1), (q1, theta1), (q1, theta0)
    if clockwise:
        if theta1 > theta0:
            theta0 += 2*np.pi

    else:
        if theta0 > theta1:
            theta1 += 2*np.pi

    subtended_angle = np.abs(theta1 - theta0)
    n_points = int(subtended_angle*180*n_points_per_degree/np.pi)+1

    angles = np.linspace(theta0, theta1, n_points)

    xs = np.concatenate((q0*np.cos(angles), q1*np.cos(angles[::-1])))
    ys = np.concatenate((q0*np.sin(angles), q1*np.sin(angles[::-1])))

    return np.array((xs, ys)).T


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    xy = wedge(0.3, 0.6, 2, 3)

    plt.plot(xy[:,0], xy[:,1])
    plt.show()

