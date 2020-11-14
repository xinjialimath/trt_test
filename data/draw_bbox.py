import cv2

img = cv2.imread('./test.jpg')
save_name = ('./results.jpg')

prob = 0.734425
x = 0
y = 66
w = 91
h = 41
cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0))
cv2.imwrite(save_name, img)