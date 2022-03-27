def SD_array(imageL, imageR, d_minimum, d_maximum):
    # initialization of the array of "squared differences" for different shifts 
    d = 1 + d_maximum - d_minimum
    SD = np.zeros((d,np.shape(imageL)[0],np.shape(imageL)[1])) 
    # couldn't figure out how to do this without a for loop
    for i in range(0, d):
        shifted_imageR = np.roll(imageR, d_minimum + i, axis=1)
        D = imageL[:,:, 0:3] - shifted_imageR[:,:, 0:3]
#         SD[i] = np.sqrt(np.sum(np.square(D),axis=2))
        SD[i] = np.sum(np.square(D),axis=2)
    return SD

def integral_image(img):
    integral_img = np.zeros(img.shape)
    m,n = img.shape
    
    # base cases 
    integral_img[0,0] = img[0, 0]
    for i in range(1, m):
        integral_img[i,0] = integral_img[i-1,0]  + img[i, 0]
    for j in range(1, n):
        integral_img[0, j] = integral_img[0, j-1]  + img[0, j]
        
    for i in range(1,m):
        for j in range(1,n):
            integral_img[i, j] = integral_img[i-1, j] + integral_img[i, j-1] - integral_img[i-1, j-1] + img[i, j]
    return integral_img

INFTY = np.inf

def windSum(img, window_width):
    s = integral_image(img)

    # half window_width
    half = window_width//2
    half_plus_odd = half + window_width % 2
    s = np.pad(s, (window_width, window_width), constant_values=0)

    br = np.roll(s, (-half, -half), axis=(0,1))
    bl = np.roll(s, (-half, half_plus_odd) , axis=(0,1))
    tl = np.roll(s, (half_plus_odd, half_plus_odd) , axis=(0,1))
    tr = np.roll(s, (half_plus_odd, -half) , axis=(0,1))

    # windSum for inside img (boundary values are wrong)
    wS_inside = br - bl - tr + tl
    wS_inside = wS_inside[window_width:-window_width, window_width:-window_width] 

    # pad boundaries with infty
    wS = INFTY * np.ones(img.shape)
    half_minus_odd = half - ((window_width + 1) % 2) 
    
    wS[half_minus_odd:img.shape[0]-half, half_minus_odd:img.shape[1]-half] = t[half_minus_odd:img.shape[0]-half, half_minus_odd:img.shape[1]-half]
    return test

def SSDtoDmap(SSD_array, d_minimum, d_maximum):
    dMap = np.argmin(SSD_array,axis=0) + d_min
    dMap = np.where(dMap == INFTY, 0, dMap)
    return dMap

#  I never finished viterbi, someone should upload theirs here