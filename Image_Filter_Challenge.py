import numpy as np
import matplotlib.pyplot as plt 
import cv2

# convert pixel to image
def set_size(width, height, ax=None):
    """ w, h: width, height in inches """
    w = width; h = height
    if not ax: ax=plt.gca()  # gets current axes
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def show_image(rgbdata, hgt_in=5.42, wid_in=5.42 ):
    """ shows an image whose pixels are in rgbdata 
        note:  rgbdata is a list-of-rows-of-pixels-of-rgb values, _not_ a filename!
        hgt_in is the desired height (in inches)
        wid_in is the desired width (in inches)
        _but_ the library will not change the aspect ratio (takes the smaller)
        by default, the hgt_in and wid_in are 5.42 in.
    """
    fig, ax = plt.subplots()           # obtains the figure and axes objects
    im = ax.imshow(rgbdata)            # this is matplotlib's call to show an image 
    set_size(width=wid_in, height=hgt_in)  # matplotlib will maintain the image's aspect ratio
    # ax.axis('off')   # turns off the axes (in units of pixels)
    plt.show()




# Load the file
data = np.load('retinamnist.npz', allow_pickle = True)
L = data.files
for i in L:
    print(i)
    
train_ims = data['train_images']
# do the same for each of these other five:
print("len(train_ims) is", len(train_ims))

first_image = train_ims[0]  
print("shape of first_image is", first_image.shape)




# Test
row0 = first_image[0]
print("len(row[0]) is", len(row0))

col0 = row0[0]  # = first_image[0][0]
print("col0 should be one pixel:", col0)

r, g, b = col0    # individual red green and blue components
print("r,g,b of the first pixel in the first row:", r,g,b)



# Filter1: Grayscale
rgbdata = first_image
def new_pixel_another( rgbpixel ):
    """ rgbpixel should be in the form [r,g,b]
        returns [newred, newgreen, new blue],
        based on their old versions!
    """
    [r,g,b] = rgbpixel
    gray = int(0.21*r + 0.72*g +0.07*b)
    if gray>225:
        gray = 225
    newr = gray
    newg = gray
    newb = gray
    return [newr, newg, newb]

def grayscale( rgbdata ):
    """ transforms an image, returning a new image 
        applies the function new_pixel to each pixel
    """
    new_rgbdata = [[ new_pixel_another(pix) for pix in row] for row in rgbdata]
    return new_rgbdata

new_rgbdata = grayscale(rgbdata)

show_image(rgbdata) # show old
show_image(new_rgbdata)   # show new



# Filter 2: Binarize
def binarize(rgbdata):
    """ transforms an image, returning a new image 
        applies the function new_pixel to each pixel
    """
    thresh = input("What is your brightness value between 0 to 225?")
    
    def binarize_pixel(rgbpixel, thresh):
    
        [r,g,b] = rgbpixel
        if int(0.21*r + 0.72*g +0.07*b) < int(thresh):
            return [225,225,225]
        else:
            return [0,0,0]
    
    new_rgbdata1 = [[ binarize_pixel(pix,thresh) for pix in row] for row in rgbdata]
    return new_rgbdata1

new_rgbdata1 = binarize(rgbdata)
show_image(new_rgbdata1)   # show new



# Filter 3: Gaussian Filter
new_image = cv2.GaussianBlur(first_image, (3, 3),0)
plt.figure(figsize=(11,6))
plt.subplot(121), plt.imshow(cv2.cvtColor(first_image, cv2.COLOR_HSV2RGB)),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)),plt.title('Gaussian Filter')
plt.xticks([]), plt.yticks([])
plt.show()
    