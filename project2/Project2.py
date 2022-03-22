import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
import cv2
from sklearn import svm


# loads all the images in the folder and returns
# an array of the image pixel values,
# an array of the image classifications,
# an array of the image file names
def load_images(path):
    img_arr = []
    class_arr = []
    file_arr = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            img_arr.append(img)
            if filename[0:3] == "024":
                class_arr.append(0)
            elif filename[0:3] == "051":
                class_arr.append(1)
            elif filename[0:3] == "251":
                class_arr.append(2)
            else:
                class_arr.append(1000)
            file_arr.append(filename)
    return img_arr, class_arr, file_arr


train_arr, train_classes, train_files = load_images("./Project2_data/TrainingDataset")

# computes the feature descriptors for each training image

sift = cv2.SIFT_create()
fd_arr = []
d_arr = []
for img in train_arr:
    [f, d] = sift.detectAndCompute(img, None)
    fd_arr.append([f, d])
    d_arr.append(d)

# puts all the feature descriptors into a single 2D array
d_arr2 = np.array(d_arr[0])
for i in range(1, len(d_arr)):
    d_arr2 = np.append(d_arr2, d_arr[i], axis=0)

# does kmeans on the feature descriptors
kmean = KMeans(n_clusters=100, random_state=0)
kmean2 = MiniBatchKMeans(n_clusters=100, random_state=0)
train_cluster = kmean2.fit(d_arr2)

# for each image, make a histogram
bins = np.arange(0, 101)
hist_arr = []
for i in range(0, len(d_arr)):
    predicts = train_cluster.predict(d_arr[i])
    hist, bins_arr = np.histogram(predicts, bins, density=True)
    hist_arr.append(hist)

# testing data part, basically repeat above steps
test_arr, test_classes, test_files = load_images("./Project2_data/TestingDataset")

# get features and descriptors
test_fd_arr = []
test_d_arr = []
for img in test_arr:
    [f, d] = sift.detectAndCompute(img, None)
    test_fd_arr.append([f, d])
    test_d_arr.append(d)

# put all descriptors in a 2d array
test_d_arr2 = np.array(d_arr[0])
for i in range(1, len(d_arr)):
    test_d_arr2 = np.append(d_arr2, d_arr[i], axis=0)

# kmeans + make histograms
test_cluster = kmean2.fit(d_arr2)
test_hist_arr = []
for i in range(0, len(test_d_arr)):
    predicts = test_cluster.predict(test_d_arr[i])
    hist, bins_arr = np.histogram(predicts, bins, density=True)
    test_hist_arr.append(hist)

plt.hist(test_hist_arr[0], bins=100)
plt.show()

# fit kneighbors on the training histograms, train_classes is the array of target values
# then do kneighbors on the test histograms
kneigh = KNeighborsClassifier(n_neighbors=1)
kneigh.fit(hist_arr, train_classes)
test_predicts = kneigh.predict(test_hist_arr)


# compare kneighbors results with the actual classifications to get percent of correct classifications
def accuracy(actual, predicted):
    confusion = np.zeros((3, 3))
    correct = len(predicted)
    for i in range(0, len(predicted)):
        confusion[actual[i]][predicted[i]] = confusion[actual[i]][predicted[i]] + 1
        if actual[i] != predicted[i]:
            correct = correct - 1
    return (correct / len(predicted) * 100), confusion


print(accuracy(test_classes, test_predicts))

# linear svm
linear_svc = svm.SVC(kernel="linear")
linear_svc.fit(hist_arr, train_classes)
test_predicts = linear_svc.predict(test_hist_arr)
print(accuracy(test_classes, test_predicts))

# nonlinear svm
nonlinear_svc = svm.SVC(kernel="rbf")
nonlinear_svc.fit(hist_arr, train_classes)
test_predicts = nonlinear_svc.predict(test_hist_arr)
print(accuracy(test_classes, test_predicts))
