#https://osdoc.cogsci.nl/3.2/visualangle/

from math import atan2, degrees, sqrt, pow
h = 55.88 # 22 inches to Monitor diagonal in cm
d = 70 # Distance between monitor and participant in cm
r = sqrt(pow(1050,2) + pow(1680,2)) # diagonal resolution of the monitor
# Calculate the number of degrees that correspond to a single pixel. This will
# generally be a very small value, something like 0.03.
print(r)
deg_per_px = 2*degrees(atan2(.5*h, d)) / r
print('%s degrees correspond to a single pixel' % deg_per_px)
print(1/deg_per_px)


print()
print(sqrt(pow(47.39,2)+pow(29.62,2)))


'''
https://tomekloboda.net/res/research/deg-to-pix/
https://osdoc.cogsci.nl/3.2/visualangle/
https://res18h39.netlify.com/calculator

Distance from the screen (D)
700 mm 
Screen diameter(d)
22 inches
Horizontal screen resolution(sx)
1680 pixels
Vertical screen resolution(sy)
1050 pixels
Fixation radius(α)
1 degrees

Fixation radius(p)
43.32 pixels

***
45.52 from convert_degree2pix.py
'''
print()

h = 29.62
d = 70 # Distance between monitor and participant in cm
r = 1050
# Calculate the number of degrees that correspond to a single pixel. This will
# generally be a very small value, something like 0.03.
deg_per_px = 2*degrees(atan2(.5*h, d)) / r
print('%s degrees correspond to a single pixel' % deg_per_px)
print(1/deg_per_px)

print()

h = 47.39
d = 70 # Distance between monitor and participant in cm
r = 1680
# Calculate the number of degrees that correspond to a single pixel. This will
# generally be a very small value, something like 0.03.
deg_per_px = 2*degrees(atan2(.5*h, d)) / r
print('%s degrees correspond to a single pixel' % deg_per_px)
print(1/deg_per_px)

print()

# Classification of visual and linguistic tasks using eye-movement features
# Coco & Keller
h = 21*2.54 # 22 inches to Monitor diagonal in cm
d = 60 # Distance between monitor and participant in cm
r = sqrt(pow(768,2) + pow(1024,2)) # diagonal resolution of the monitor
# Calculate the number of degrees that correspond to a single pixel. This will
# generally be a very small value, something like 0.03.
print(r)
deg_per_px = 2*degrees(atan2(.5*h, d)) / r
print('%s degrees correspond to a single pixel' % deg_per_px)
print(1/deg_per_px)
