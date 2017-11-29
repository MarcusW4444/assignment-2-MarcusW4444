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
    filtername = None
    def __init__(self, image, filter_name, cutoff, order = 0):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        filter_name: the name of the mask to use
        cutoff: the cutoff frequency of the filter
        order: the order of the filter (only for butterworth
        returns"""


        self.image = image
        filtername = filter_name
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
        self.filtername = filter_name
        self.cutoff = cutoff
        self.order = order


    def get_ideal_low_pass_filter(self, shape, cutoff):
        """Computes a Ideal low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal low pass mask"""

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
                D = pow(pow(u - (sx / 2), 2) + pow((v - (sy / 2)), 2), 1 / 2)
                h = (1.0/(1.0+pow((D/cutoff),2*order)))
                newmask[u,v] = h
        
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
                D = pow(pow(u - (sx / 2), 2) + pow((v - (sy / 2)), 2), 1 / 2)
                h = 1-(1.0 / (1.0 + pow((D / cutoff), 2 * order)))
                #print((D,h))
                if(False):
                    if (D <= 0.0):
                        h = 0
                    else:
                        f = (cutoff/D)
                        e = (1.0 + pow(f, 2 * order))
                        h = (1.0 / e)

                newmask[u, v] = h
        
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
                D = pow(pow(u - (sx / 2), 2) + pow((v - (sy / 2)), 2), 1 / 2)
                h = np.exp((-(pow(D,2)))/(2*pow(cutoff,2)))
                newmask[u, v] = h

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
                D = pow(pow(u - (sx / 2), 2) + pow((v - (sy / 2)), 2), 1 / 2)
                h = 1-np.exp(-(pow(D, 2)) / (2*pow( cutoff, 2)))
                newmask[u, v] = h

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
        b = False
        mn = 0
        mx = 0
        for u in range(sx):
            for v in range(sy):
                t = image[u,v]
                if (b):
                    mn = min(t,mn)
                    mx = max(t,mx)
                else:
                    b = True
                    mn = t
                    mx = t
                #get max and min for contrast stretch


        for u in range(sx):
            for v in range(sy):
                if (mn < mx):
                    newimage[u, v] = 255-(255*((newimage[u, v]-mn)/(mx-mn))) #full contrast stretch and negative
                else:
                    newimage[u, v] = mx

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

        #Actual convolution below. Takes an eternity to process...
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
        mx = np.max(im)
        mn = np.min(im)

        im = np.log(np.abs(im))


        mx = np.max(im)
        mn = np.min(im)
        #print((mx, mn))
        for u in range(im.shape[0]):
            for v in range(im.shape[1]):
                val = im[u, v]


                im[u,v] =\
                    (255*((val - mn) / (mx - mn)))

        im = im * (255 / max(1, np.max(im)))
        return im
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
        msk = np.zeros(fshift.shape)

        if self.filtername == 'ideal_l':
            print("ideal_l filter")
            msk = self.get_ideal_low_pass_filter(fshift.shape, self.cutoff)
        elif self.filtername == 'ideal_h':
            print("ideal_h filter")
            msk = self.get_ideal_high_pass_filter(fshift.shape, self.cutoff)
        elif self.filtername == 'butterworth_l':
            print("butterworth_l filter")
            msk = self.get_butterworth_low_pass_filter(fshift.shape, self.cutoff, self.order)
        elif self.filtername == 'butterworth_h':
            print("butterworth_h filter")
            msk = self.get_butterworth_high_pass_filter(fshift.shape, self.cutoff, self.order)
        elif self.filtername == 'gaussian_l':
            print("gaussian_l filter")
            msk = self.get_gaussian_low_pass_filter(fshift.shape, self.cutoff)
        elif self.filtername == 'gaussian_h':
            print("gaussian_h filter")
            msk = self.get_gaussian_high_pass_filter(fshift.shape, self.cutoff)
        else:
            print("unknown filter: " + self.filtername)
            msk = self.get_ideal_low_pass_filter(fshift.shape, self.cutoff)
        umsk = msk
        print((np.min(msk), np.max(msk)))
        #4. filter the image frequency based on the mask (Convolution theorem)
        #print("Convolving")
        #co = self.convolve(fshift,msk)
        #print("Done Convolving")

        #5. compute the inverse shift

        appliedfilter = fshift*msk
        ishift = np.fft.ifftshift(appliedfilter)

        #6. compute the inverse fourier transform
        img = np.fft.ifft2(ishift)
        #7. compute the magnitude



        #8. You will need to do a full contrast stretch on the magnitude and depending on the algorithm you may also need to
        #take negative of the image to be able to view it (use post_process_image to write this code)
        #co = np.log(np.abs(co))


        #Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        #filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey scale full contrast stretch and dtype=uint8


        magnitudes = np.log(np.abs(img))
        magnitudes = magnitudes*(255/max(1,np.max(magnitudes)-np.min(magnitudes))) #full contrast stretch

        #appliedfilter = np.abs(appliedfilter)
        #appliedfilter = appliedfilter * (255 / max(1, np.max(appliedfilter) - np.min(appliedfilter)))  # full contrast stretch
        #co = co * (255 / max(1, np.max(co)))  # full contrast stretch
        #img = img * (255 / max(1, np.max(img)))  # full contrast stretch
        #negative = self.post_process_image(magnitudes)
        msk = msk * (255 / max(1, np.max(msk)))
        img = np.log(np.abs(img))
        img = img * (255 / max(1, np.max(img)))
        appliedfilter = np.log(np.abs(appliedfilter ))
        appliedfilter = appliedfilter * (255 / max(1, np.max(appliedfilter)))
        #co = np.log(np.abs(co))
        #co = co * (255 / max(1, np.max(co)))
        np.log(np.abs(appliedfilter))
        return [np.uint8(self.normalize(fshift)), appliedfilter, np.uint8(self.normalize(magnitudes))]
