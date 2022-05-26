import numpy as np
# import math
import matplotlib.pyplot as plt


OUTPUT_MASK_FILE = 'samples/mask_insole_L.npy'


CONN_TRIM = 22 + 5  # mm

# medial-lateral direction, from left to right
CONN_H = [
367.7, 356.13, 346.53, 337.85, 330.06, 322.91, 315.7, 309.11, 302.85,
297.19, 291.52, 282.67, 271.73, 266.9, 261.82, 256.44, 251.27, 245.61,
239.93, 233.77, 227.83, 221.97, 215.57, 209.66, 205.41, 204.51, 205.47
]

SENSOR_H = [
50.14, 66.02, 77.68, 86.0, 91.97, 96.4, 99.37, 101.16, 101.25,
100.33, 98.95, 96.3, 93.09, 89.89, 86.97, 83.61, 80.99, 78.44,
76.3, 74.83, 74.43, 73.71, 72.27, 70.27, 67.02, 64.8, 62.22
]

# lateral-posterior direction, from top to bottom
CONN_V = [210.2, 205.0, 202.42, 200.52, 200.0, 200.69, 202.79, 206.28, 212.21]

SENSOR_V = [288.12, 286.96, 285.3, 283.79, 282.03, 280.56, 278.17, 275.09, 270.57]



def get_mask(conn_h, conn_v, sensor_h, sensor_v, conn_trim):
    """Get mask of insole sensor.
    e.g.      fore-foot
          (0,0) __ __ (0,2)
               |__|__|
               |__|__|          
          (2,0)       (2,2)
                heel
    Args
        conn_h (list[float]):
        conn_v (list[float]):
        sensor_h (list[float]):
        sensor_v (list[float]):
        conn_trim (float):
    Returns
        (np.array[float]): Mask
    """
    num_h = len(conn_h)
    num_v = len(conn_v)
    print('Number of electrodes (col, row): ({}, {})'.format(num_v, num_h))

    mask = []
    for i in range(num_h):
        mask_row = []
        for j in range(num_v):
            len_h = conn_h[i] - conn_trim + (j+1)*sensor_h[i]/(num_v+1)
            len_v = conn_v[j] - conn_trim + (num_h-i)*sensor_v[j]/num_h
            mask_row.append(len_h+len_v)
        mask.append(mask_row)

    return np.asarray(mask)


def normalize_mask(mask):
    """ """
    min = np.min(mask)
    ret = mask/min
    return ret


def rotate_clockwise(mask):
    """ """
    ret = np.zeros_like(mask.T)
    print(mask.shape, ret.shape)  # (27,9), (9,27)

    for i in range(mask.shape[1]):  # 27
        for j in range(mask.shape[0]):  # 9
            ret[i, mask.shape[0]-j-1] = mask[j, i]
    return ret


def main():

    mask = get_mask(CONN_H, CONN_V, SENSOR_H, SENSOR_V, CONN_TRIM)
    print(mask[0,0], mask[0,8])
    print(mask[26,0], mask[26,8])

    mask = normalize_mask(mask)
    
    mask = rotate_clockwise(mask)

    # np.save(OUTPUT_MASK_FILE, mask)

    plt.imshow(mask)
    plt.colorbar()
    plt.show()


if __name__=='__main__':
    main()