Marcus Washington

If there are any unexpected problems with running the code, please contact me at maw.rawcat4@gmail.com

Data Image Processing Assignment 2 Report

1. DFT
Starting by initiating a 15x15 image matrix of pure noise. The contents of the image are printed to the console, showing
a 15x15 array of intensity values from 0-255. The image is then transformed through a sequence of hand-made DFT algorithms.
For all of the algorithms, each one evaluates every cell in the image/transform by iterating through all of the cells in
the structure. So for an image of length and width of N, an algorithm may make up to N^4 iterations

-forward_transform
This function takes the image and converts it to its Discrete Fourier Transform by calculating a summation of the intensity
values of each cell in the image multiplied by the intensity of the current pixel and the Real (cos) and Imaginary (-1j * sin)terms
It results in a series of positive and negative scientific numbers for some particular reason.

-inverse_transform
The purpose of this function is to take an existing DFT as an input and convert it back to its original image form.
While the values do manage to get reigned into the proper range, the result does not exactly seem precisely like the original or 1 to 1.

-magnitude
This function is meant to look at the values of a DFT and determine the magnitude of their components. Looking at the results,
it seems that most magnitudes of the image cells are quite level sitting near the value of N/2 with smaller outliers along the left edge.

-discrete_cosine_tranform
Similar to the forward_transform function, this function calculates the Discrete Cosine transform of the original image.
Due to the noisy nature of the original image, both the forward transform and cosine transform functions
do not really yield any discernible pattern or behavior as opposed to looking at the transforms of a common photo.


2. Frequency Filtering
Frequency Filtering is the process of converting an image into a fourier transform, directly applying the filter or mask
to the cells of the transform by means of convolution and then convert the transform back to the original picture but
without the frequency patterns that have been masked out. The following functions determine the contents of the mask.

-get_ideal_low_pass_filter
This filter is discrete, meaning that the range of values in the filter strictly consists of 0 and 1 only; no intermediary values
Despite how simple this filter seems, it is remarkably effective at removing low levels of noise from the image.
It definitely is an ideal means of noise removal. (ha!)
However, some of the sharper details could be mistaken for noise and ultimately end up being washed out by the filter.

-get_gaussian_low_pass_filter
This filter uses a gradient curve which is much more lenient on the miniscule but sharp texture details that commonly appear on
continuous surfaces. This filter may be more favorable if small details are critical to the value/meaning of the image.

-get_butterworth_low_pass_filter
This filter features a gradient similar to the gaussian filter but possesses a slightly steeper curve, meaning that it would be able
to smoothen out noise slightly lower frequencies of noise compared to the gaussian filter. Theoretically, this filter would also be capable of
eradicating sharper, more-sudden spikes of noise pixels

-get_ideal_high_pass_filter, get_butterworth_high_pass_filter, get_gaussian_high_pass_filter
High pass filters have shown that they can effectively isolate the tiniest quantities of noise that their low pass counterparts would typically filter out.
One could use these filters for less trivial motivations that require more sophisticated and complex manipulation of the artifacts collected by these filters.
One of the simpler uses for the results obtained from these filters is to simply subtract the collected noise from the original image smoothen it out;
more advanced cases will require more processing to be done on this noise to achieve a very specific solution.


Issues:
-For some accidental reason, performing the inverse transform has caused the image to overlap unto itself;
that is most caused by a poor choice of Numpy function on my part.
Even with this issue, the impacts of the filters are still noticeable in side by side comparisons.

-When either the Order or Cutoff of the filters is too low on gradient filters like Butterworth and Gaussian,
the gradient in the filter ends up cycling multiple times depending on the value.

Mathematical Errors cause the masks for the filters to become rather noisy, generating some inappropriate spikes on random pixels
However, the impact of these errors is not very prominent since these spikes are rather miniscule on the mask,
meaning that there are not enough pixels affected by the spike to make any noticeable difference


