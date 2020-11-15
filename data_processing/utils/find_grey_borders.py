from PIL import Image

'''From DIDEC paper: The eye tracker had a sampling rate of 250 Hz. 
The stimulus materials were displayed on a 22 inch P2210 Dell monitor,
with the resolution set to 1680 x 1050 pixels. 
The images were resized to 1267 x 950 pixels (without changing the aspect ratio), 
surrounded by grey borders. These borders were required because eye-tracking measurements 
outside the calibration area (i.e., in the most peripheral areas of the screen) are not reliable.
 The viewing distance was 70 cm.'''

# size of the whole screen
width_s = 1680
height_s = 1050

# size of the original image in the gray borders
width_o = 1267
height_o = 950


orig_img = Image.open('../data/images/61514.jpg', 'r')

width, height = orig_img.size
print(orig_img.size) #1680, 1050
pix = orig_img.load()
grey_RGB = (129,129,129)

for i in range(width_o):
    for j in range(height_o):

        if pix[i,j] == grey_RGB:
            print('GREY IN IMAGE', i,j)

#to see if the image has 129,129,129 grey in it
#THERE WAS NO GREY IN THIS IMAGE


#get where the grey borders end in the transformed image
temp_img = Image.open('../data/images_bordered/61514.bmp', 'r')
width, height = temp_img.size
print(temp_img.size) #1680, 1050
pix = temp_img.load()
grey_colour = pix[width-1,height-1] #0,0 width,height

img_xs = []
img_ys = []

for i in range(width_s):
    for j in range(height_s):

        if pix[i,j] != grey_colour:
            #print(i,j)
            img_xs.append(i)
            break

for j in range(height_s):
    for i in range(width_s):

        if pix[i,j] != grey_colour:
            #print(i,j)
            img_ys.append(j)
            break

grey_xs = set(img_xs)
grey_ys = set(img_ys)

print(len(img_xs), img_xs) #(1267 actual size)
print(len(img_ys), img_ys) #(950 actual size)

#indices start from 0
# actual image's coordinates:
print(min(img_xs), max(img_xs)) # 206 1472
print(min(img_ys), max(img_ys)) # 50 999

assert max(img_xs) - min(img_xs) + 1 == 1267
assert max(img_ys) - min(img_ys) + 1 == 950

'''
GREY REGION
x range 
0-205 1473-1679
y range 
0-49 1000-1049
'''