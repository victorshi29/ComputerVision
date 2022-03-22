# Do not import any additional modules
import numpy as np
from PIL.Image import open
import matplotlib.pyplot as plt

### Load, convert to grayscale, plot, and resave an image
I = np.array(open('Iribe.jpg').convert('L')) / 255

plt.imshow(I, cmap='gray')
plt.axis('off')
plt.show()

plt.imsave('test.png', I, cmap='gray')


### Part 1
def sumarr(arr):
    sum = 0
    for i in arr:
        for j in i:
            sum = sum + j
    return sum


def gausskernel(sigma):
    x = np.linspace(-1, 1, 3 * sigma)
    y = np.linspace(-1, 1, 3 * sigma)
    x2d, y2d = np.meshgrid(x, y)
    kernel = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    return kernel / sumarr(kernel)


def weightedsum(weights, values):
    sum = 0
    for row in range(0, len(weights)):
        for col in range(0, len(weights[0])):
            sum = sum + weights[row][col] * values[row][col]
    return sum


def addpadding(I, radius):
    newarr = np.zeros((len(I) + 2 * radius, len(I[0]) + 2 * radius))
    for row in range(0, len(I)):
        for col in range(0, len(I[0])):
            newarr[row + radius][col + radius] = I[row][col]
    return newarr


def myfilter(I, h):
    knlrad = np.max([len(h), len(h[0])]) // 2
    L = I.copy()
    K = addpadding(I, knlrad)
    for row in range(0, len(I)):
        for col in range(0, len(I[0])):
            subarray = K[row:row + len(h), col:col + len(h[0])]
            L[row][col] = weightedsum(subarray, h)
    return L


h1 = np.array([[-1 / 9, -1 / 9, -1 / 9], [-1 / 9, 2, -1 / 9], [-1 / 9, -1 / 9, -1 / 9]])
h2 = np.array([[-1, 3, -1]])
h3 = np.array([[-1], [3], [-1]])

### Part 2
Sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
Sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


def derivatives(I):
    dx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    dy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filter_dx = myfilter(I, dx)
    filter_dy = myfilter(I, dy)
    return filter_dx, filter_dy


def gradient(dx, dy):
    grad = np.zeros((len(dx), len(dx[0])))
    for row in range(0, len(dx)):
        for col in range(0, len(dx[0])):
            x = dx[row][col]
            y = dy[row][col]
            grad[row][col] = np.power(np.power(x, 2) + np.power(y, 2), 0.5)
    return grad


def grad_angle_deriv_helper(val):
    if np.pi / 8 >= val >= -np.pi / 8:
        return 0
    elif np.pi / 8 <= val <= 3 * np.pi / 8:
        return 45
    elif (3 * np.pi / 8 <= val <= np.pi / 2) or (-np.pi / 2 <= val <= -3 * np.pi / 8):
        return 90
    elif -3 * np.pi / 8 <= val <= -np.pi / 8:
        return 135
    else:
        return 90


def grad_angle(dx, dy):
    angle = np.zeros((len(dx), len(dx[0])))
    for row in range(0, len(dx)):
        for col in range(0, len(dx[0])):
            if dx[row][col] == 0:
                angle[row][col] = np.pi / 2
            else:
                angle[row][col] = np.arctan(dy[row][col] / dx[row][col])
    return angle


def grad_angle_deriv(I):
    I = I.copy()
    angle_deriv = np.zeros((len(I), len(I[0])))
    for row in range(0, len(I)):
        for col in range(0, len(I[0])):
            angle_deriv[row][col] = grad_angle_deriv_helper(I[row][col])
    return angle_deriv


def edge_thin_helper1(pixel, grad_arr):
    if pixel == 90:
        return grad_arr[1][1] > grad_arr[2][1] and grad_arr[1][1] > grad_arr[0][1]
    elif pixel == 45:
        return grad_arr[1][1] > grad_arr[2][2] and grad_arr[1][1] > grad_arr[0][0]
    elif pixel == 0:
        return grad_arr[1][1] > grad_arr[1][2] and grad_arr[1][1] > grad_arr[1][0]
    elif pixel == 135:
        return grad_arr[1][1] > grad_arr[2][0] and grad_arr[1][1] > grad_arr[0][2]


def edge_thin(grad, grad_deriv):
    L = np.copy(grad)
    K = addpadding(L, 1)
    for row in range(0, len(grad)):
        for col in range(0, len(grad[0])):
            subarr = K[row:row + 3, col:col + 3]
            if edge_thin_helper1(grad_deriv[row][col], subarr):
                L[row][col] = grad[row][col]
            else:
                L[row][col] = 0
    return L


def threshold(I, grad_arr, t_low1, t_high1):
    L = np.zeros((len(I), len(I[0])))
    from scipy.ndimage.measurements import label
    labels, numfeatures = label(I)
    for row in range(0, len(I)):
        for col in range(0, len(I[0])):
            if grad_arr[row][col] > t_high1:
                L[row][col] = 255
            else:
                label = labels[row][col]
                row2, col2 = np.where((labels == label))# & (grad_arr >= t_low1) & (grad_arr <= t_high1))
                isedge = False
                for i in range(0, len(row2)):
                    if grad_arr[row2[i]][col2[i]] > t_high1:
                        isedge = True
                        break
                if isedge:
                    for j in range(0, len(row2)):
                        L[row2[j]][col2[j]] = 255
    return L


def myCanny(I, sigma=1, t_low=.5, t_high=0.55):
    # Smooth with gaussian kernel
    kernel = gausskernel(sigma)
    gauss_img = myfilter(I, kernel)
    plt.imshow(gauss_img, cmap='gray')
    plt.axis('off')
    plt.show()

    # Find img gradients
    x, y = derivatives(gauss_img)
    grad = gradient(x, y)

    plt.imshow(x, cmap='gray', vmin=0)
    plt.axis('off')
    plt.show()
    plt.imshow(y, cmap='gray', vmin=0)
    plt.axis('off')
    plt.show()
    plt.imshow(grad, cmap='gray', vmin=0)
    plt.axis('off')
    plt.show()

    theta = grad_angle(x, y)
    theta_deriv = grad_angle_deriv(theta)

    # Thin edges
    print("thinning")
    thinned = edge_thin(grad, theta_deriv)
    plt.imshow(thinned, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.show()

    # Hystersis thresholding
    from scipy.ndimage.measurements import label
    print("final image")
    finished = threshold(thinned, thinned, t_low, t_high)
    plt.imshow(finished, cmap='gray', vmin=0)
    plt.axis('off')
    plt.show()

    return finished

edges = myCanny(I, sigma=1, t_low=.3, t_high=0.5)
#plt.imshow(edges, interpolation='none')
#s = myfilter(I,h1)
#plt.imshow(s, cmap='gray', vmin=0)
#plt.show()
