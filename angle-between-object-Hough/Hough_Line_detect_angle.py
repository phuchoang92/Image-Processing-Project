import cv2
import math
import numpy as np

norm = np.linalg.norm

img = cv2.imread("P:\\Projects\\PythonProjects\\ProjectsAI\\ImageProcessing\\images\\img2.png_noise.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("blur", gray)

blur = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
cv2.imshow("blur", blur)

gamma_corrected = np.array(255 * (blur / 255) ** 0.15, dtype='uint8')
cv2.imshow("Gamma Correction", gamma_corrected)

thresh = cv2.adaptiveThreshold(gamma_corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imshow("After Threshold", thresh)

erose = 255 - thresh
cv2.imshow("Negative Image", erose)

lines = cv2.HoughLinesP(erose, 1, np.pi / 180, 200, minLineLength=200, maxLineGap=40)
distances = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    p3 = np.array([0, 0])
    distance = np.abs(norm(np.cross(p2 - p1, p1 - p3))) / norm(p2 - p1)
    distances.append(distance)
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

vectors = []
for distance, line in zip(distances, lines):
    if distance == max(distances) or distance == min(distances):
        x1, y1, x2, y2 = line[0]
        vectors.append(np.array([x1 - x2, y1 - y2]))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

a1, a2, b1, b2 = vectors[0][0], vectors[1][0], vectors[0][1], vectors[1][1]
cos = (abs(a1 * a2 + b1 * b2)) / (math.sqrt(a1 ** 2 + b1 ** 2) * math.sqrt(a2 ** 2 + b2 ** 2))

angle = math.degrees(np.arccos(cos))
if angle > 90:
    angle = 180 - angle

label = "  Angle: " + str(round(angle)) + " degree"
#cv2.putText(img, label, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

cv2.imshow('Result Image', img)
cv2.waitKey(0)

'''bilateral= cv2.bilateralFilter(img, 9, 75, 75)
cv2.imshow("bilateral", bilateral)'''
'''edges = cv2.Canny(gray, 100, 150, apertureSize=3)
cv2.imshow("edge", edges)'''
# thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
