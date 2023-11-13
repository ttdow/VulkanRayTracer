import cv2 as cv

img1 = cv.imread("gold_foil_metalness.png", cv.IMREAD_UNCHANGED)
img2 = cv.imread("gold_foil_roughness.png", cv.IMREAD_UNCHANGED)

#img1 = cv.cvtColor(img1, cv.COLOR_RGB2RGBA)
#img2 = cv.cvtColor(img2, cv.COLOR_RGB2RGBA)

#cv.imwrite("gold_foil_metalness.png", img1)
#cv.imwrite("gold_foil_roughness.png", img2)

b1, g1, r1, a1 = cv.split(img1)
b2, g2, r2, a2 = cv.split(img2)

new_image = cv.merge([b2, g2, r1, a2])

cv.imwrite("gold_foil_combined.png", new_image)

#image1 = Image.open("gold_foil_albedo.png")
#image2 = Image.open("gold_foil_roughness.png")

#print(image1)

#R = Image.Image.split(image1)[0]

#R.save("test.png")

#r1 = image1.split()
#r2 = image2.split()

#print(len(r2))

#new_image = Image.merge("RGB", (r1[0], r1[0], r1[0]))

#new_image.save("gold_foil_combined.png")