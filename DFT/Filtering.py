# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv

import cv2
import numpy as np
from matplotlib import pyplot as plt

class Filtering:
    image = None
    filter = None
    cutoff = None
    order = None

    def __init__(self, image, filter_name, cutoff, order = 0):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        filter_name: the name of the mask to use
        cutoff: the cutoff frequency of the filter
        order: the order of the filter (only for butterworth
        returns"""
        self.image = image
        if filter_name == 'ideal_l':
            self.filter = self.get_ideal_low_pass_filter
        elif filter_name == 'ideal_h':
            self.filter = self.get_ideal_high_pass_filter
        elif filter_name == 'butterworth_l':
            self.filter = self.get_butterworth_low_pass_filter
        elif filter_name == 'butterworth_h':
            self.filter = self.get_butterworth_high_pass_filter
        elif filter_name == 'gaussian_l':
            self.filter = self.get_gaussian_low_pass_filter
        elif filter_name == 'gaussian_h':
            self.filter = self.get_gaussian_high_pass_filter

        self.cutoff = cutoff
        self.order = order


    def get_ideal_low_pass_filter(self, shape, cutoff):
        """Computes a Ideal low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal low pass mask"""
        print("Computing ideal low pass pask")
        sx = shape[0]
        sy = shape[1]
        newmask = np.zeros((sx, sy))
        for u in range(sx):
            for v in range(sy):
                if (pow(pow(u-(sx/2),2)+ pow((v-(sy/2)),2),1/2)<=cutoff):
                    newmask[u, v] = 1
                else:
                    newmask[u, v] = 0




        return newmask


    def get_ideal_high_pass_filter(self, shape, cutoff):
        """Computes a Ideal high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask

        sx = shape[0]
        sy = shape[1]
        newmask = np.zeros((sx, sy))
        for u in range(sx):
            for v in range(sy):
                if (pow(pow(u - (sx / 2), 2) + pow((v - (sy / 2)), 2), 1 / 2) <= cutoff):
                    newmask[u, v] = 0.0
                else:
                    newmask[u, v] = 1.0
        
        return newmask

    def get_butterworth_low_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth low pass mask"""
        sx = shape[0]
        sy = shape[1]
        newmask = np.zeros((sx, sy))
        for u in range(sx):
            for v in range(sy):
                newmask[u,v] = .5
        
        return newmask

    def get_butterworth_high_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        sx = shape[0]
        sy = shape[1]
        newmask = np.zeros((sx, sy))
        for u in range(sx):
            for v in range(sy):
                newmask[u,v] = .5
        
        return newmask

    def get_gaussian_low_pass_filter(self, shape, cutoff):
        """Computes a gaussian low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian low pass mask"""
        sx = shape[0]
        sy = shape[1]
        newmask = np.zeros((sx, sy))
        for u in range(sx):
            for v in range(sy):
                newmask[u, v] = .5

        return newmask


    def get_gaussian_high_pass_filter(self, shape, cutoff):
        """Computes a gaussian high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        sx = shape[0]
        sy = shape[1]
        newmask = np.zeros((sx, sy))
        for u in range(sx):
            for v in range(sy):
                newmask[u, v] = .5

        return newmask

    def post_process_image(self, image):
        """Post process the image to create a full contrast stretch of the image
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        1. Full contrast stretch (fsimage)
        2. take negative (255 - fsimage)
        """

        sx = image.shape[0]
        sy = image.shape[1]

        newimage = np.zeros((sx, sy))
        mn = 255
        mx = 0
        for u in range(sx):
            for v in range(sy):
                t = newimage[u,v]
                mn = min(t,mn)
                mx = max(t,mx)
                #get max and min for contrast stretch

        if (mn < mx):
            for u in range(sx):
                for v in range(sy):
                    newimage[u, v] = 255-(255*((newimage[u, v]-mn)/(mx-mn))) #full contrast stretch and negative

        return newimage

    def convolve(self,image,mask):

        sx = image.shape[0]
        sy = image.shape[1]
        mx = mask.shape[0]
        my = mask.shape[1]
        midx = int(mask.shape[0]/2)
        midy = int(mask.shape[1]/2)

        padx = (midx*2)
        pady = (midy * 2)
        newimage = np.zeros((sx, sy))
        paddedimage = np.zeros((sx+padx, sy+pady))
        for u in range(sx):
            for v in range(sy): #assign values unto padded image
                paddedimage[(midx+u),(midy+v)] = image[u,v]


        for u in range(sx):
            for v in range(sy):
                t = image[u,v]*(mask[u,v])
                newimage[u,v] = t

        #for u in range(sx):
            #for v in range(sy):
                #t = 0
                #for i in range(mx):
                    #for j in range(my):  # assign values unto padded image
                        #t = t + (paddedimage[u+i,v+j]*mask[i,j])
                #t = (t / (mx * my))
                #newimage[u,v] = t

        return newimage

    def normalize(self,im):
        m = np.log(np.abs(im))
        m = m * (255 / max(1, np.max(m)))
        return m
    def filtering(self):
        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of DFT, magnitude of filtered DFT        
        ----------------------------------------------------------
        You are allowed to used inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv
        Steps:
        """
        #1. Compute the fft of the image
        f = np.fft.fft2(self.image)
        #2. shift the fft to center the low frequencies
        fshift = np.fft.fftshift(f)

        #3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape, cutoff, order)
        msk = self.filter(fshift.shape, self.cutoff)
        print(msk.shape)
        #if (filter == self.get_ideal_low_pass_filter):
            #msk = self.get_ideal_low_pass_filter(fshift.shape,self.cutoff)
        #elif (filter == self.get_ideal_high_pass_filter):
            #msk = self.get_ideal_high_pass_filter(fshift.shape, self.cutoff)
            #Butterworth
            #Gaussian

        #4. filter the image frequency based on the mask (Convolution theorem)
        print("Convolving")
        co = self.convolve(fshift,msk)
        print("Done Convolving")
        #5. compute the inverse shift
        ishift = np.fft.ifftshift(co)

        #6. compute the inverse fourier transform
        img = np.fft.ifft2(ishift)
        #7. compute the magnitude



        #8. You will need to do a full contrast stretch on the magnitude and depending on the algorithm you may also need to
        #take negative of the image to be able to view it (use post_process_image to write this code)
        #co = np.log(np.abs(co))


        #Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        #filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey scale full contrast stretch and dtype=uint8


        magnitudes = np.log(np.abs(img))
        magnitudes = magnitudes*(255/max(1,np.max(magnitudes))) #full contrast stretch
        #co = co * (255 / max(1, np.max(co)))  # full contrast stretch
        #img = img * (255 / max(1, np.max(img)))  # full contrast stretch
        negative = self.post_process_image(img)

        return [self.image, np.uint8(self.normalize(co)), np.uint8(magnitudes)]
