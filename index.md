# Group 11 Reproduction Extremely Dark Images Project
On this page we will give an overview of our reproduction efforts of the orginal research paper called “Restoring Extremely Dark Images in Real Time” (M. Lamba & K. Mitra, 2021). This reproducibilty project is made by Dries Borstlap (student number: ?), Georg Strunck (student number: ?), Koen van Vlijmen (student number: ?) and Christiaan Wiers (student number: 4715340)

## The content of our reproduction looks as follows
* Abstract 
* Introduction 
* Method 
* Results 
* Discussion
* Conculsion 

## Abstract

Text here
(what, why, method, research goals)

## Introduction and motivation

Over the years a lot of good methods for low-light image enhancement have been developed. The methods to do so thus actually exist but are often focussed on targetting restauration quality, this come at the expense of computational speed and that makes most of the existing methods for low-light image enhancement non-practical solutions [1]. In the paper "Restoring Extremely Dark Images in Real-Time" (M. Lamba & K. Mitra, 2021) a fast and memory efficient solution is presented which at the same time produces proper quality (light enhanced) images. 
In this blog post we will test the sensitivity of the researcher's model to variances applied to the input images. In other words how robust is the model created by reseachers? 

This will be done by applying variances (noise and brightness) to the input images and visually analizing the impact this may have on the quality of the model's output images. The aforementioned reults in our two research goals: 
1) Adding (or removing) noise to the input image (in the pre-processing phase) and see it's influence on the output quality of the researchers model.
2) Quantifying different brightness levels of input pictures and relating that the output quality of the researchers model. 

The reason for choosing both noise and brightness as variances to the input image is in short because we want to analyze the model's robustness because pictures may in practice also contain noise and certainly will differ greatly in brightness so that's in essence what makes this type of pre-processing research releveant. This type of research also adresses the growing concern in the deep learning community when it comes to parameter tuning and overfitting in order to improve results instead of coming up with smart / out of the box improvements. 

The reason for testing the model on the addition of noise and brightness to the input images is because first of all, both methods of variance allow us to check the model's sensitivity in an easy to control manner, by gradually increasing these variances and analyze it's influence on the output images. Secondly, noise is chosen as a variance because this addition of a random distruption to the input image allows us to analyse how well the model can handle this type of random variance. Lastly, the adjustment of brightness is chosen because it allows us to see how the researcher's model performs when the (dark) input images are brightened up.


## Method

Before going into to more detail, we try to give a structured overview of the additional code we made in order to test the sensitivy of the researcher's model to varying inputs. 
It starts with the RAW input image which is converted - in process A - to a RGB image which is plotted (step 2). In process B we use code to make the modifications to the (RGB) image. This modified RGB image is also plotted (step 3) and is later converted back to a RAW file in process C. This modified RAW images is plotted as well (step 4) in order to make sure the conversion from RGB to RAW was done succesfully. Finally this RAW file from step 4, is used process D - this is the original model from the paper - the final result is a JPEG file (step 5).

<p align="left">
  <img src="/Images/process_overview.png" height="300">  
</p>

#### (A) RAW to RGB conversion code snippet

The input used by the neural network is in the form of a raw image, which is encoded using bayer style color representation. 
This means that every data point in the image array is one color, instead of the usual three in the case of rgb.
Colors are encoded using a fixed pattern, namely:
 [R , G]
 [G , B]
This pattern repeats over the full width and height of the image. Retrieving colors can thus be done by extrapolating these values where every 2x2 block of data points as shown above is converted into 1 rgb pixel.

The bayer encoded image sadly does not work with strict values between 0 and 255, but rather a range between 0 and 65536 (16 bit), with the vast majority of numbers are being packed in the range [410,440]. A scaled distribution correction is used to transform all bayer values from the 16 bit bayer range to the 8 bit rgb spectrum.
Parameters used for this scaling (like minimum and maximum bayer range corresponding to 0 and 255 in the rgb spectrum) are fine-tuned, so that rgb converted images resemble almost perfectly the 'processed' raw images that photographic software would yield.

In the rgb spectrum, intuitive changes can now be applied to the image. Possible preprocessing adaptations are explained in the chapters below. The final feature of our code allows us to retrieve applied changes in the rgb spectrum. Using the precise inverse of operations used to convert raw (bayer) to rgb, one can convert rgb back to raw (bayer). The changes in raw (bayer) are added to the original bayer image, and used as input for the neural network.

[HIER NOG DE CODE TOEVOEGEN?]
```markdown
- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

#### (B) Adding noise code snippet

Below an excerpt of the code for adding noise shows the approach that was taking to add red, green or blue noise pixels to the image. It takes the matrix of all eg. red pixels of the image and adds random red pixels to the matrix from a normal distribution. fThe pixel location in the image is included with the location in the matrix. The same counts for green and blue. In order to make similar changes in all three colours, thus R, G and B, noise has to be added to all three matrices.

```markdown
def add_noise(color, scale):
    shape = color.shape
    noise = np.random.normal(loc=0.0, scale=scale, size=shape)
    color = np.add(color, noise).astype(int)
    return color

r = add_noise(r, 0.5)
g = add_noise(g, 0.5)
b = add_noise(b, 0.5)
```

#### (B) Adding brightness code snippet

The goal of manipulating brigthness of the test input images is to check the robustness and generalization of the model already trained by the researchers.
The original purpose of the developed method by the researchers is to 'restore' dark images. This raises the question how the model would handle input images that are already bright. Does the model overcompensate when increasing intensity of the image, degrading the quality of the image rather than improving it? 
Moreover, we are interested in the performance of the model for images that have significant variance of brighthness across the image.

In order to visually compare the performance of the model to changes in brightness, we opted for the manipulating the brightness of an existing image and feed both the original and the manipulated image to the network. This approach allows us to visually compare the output images of the same input images, with different brightness levels. 

As explained above, the encrypted RAW input images are copied and converted to RGB after which the images can be edited. Two operations to manipulate the brightness:
1. Add an integer value to the intensity of all pixels, uniform across RGB channels.
2. Add an integer value to the intensity of a local subset of the pixels, uniform across RGB channels

Please see the operations in code below

```markdown

#The function to change the brightness of an RGB image by 'value' uniformly across all pixels of the image
#Input: 'rgb' is the image as an rgb array. 'value' is the integer value that is added to the original pixel values 
#Output: returns the adjusted image as an rgb array

        def change_brightness(rgb, value):
            rgb_new = np.where((255 - rgb) < value, 255, rgb + value)
            return rgb_new

#The function to change the brightness of an RGB image by 'value',  the left hand side half of the image
#Input: 'rgb' is the image as an rgb array. 'value' is the integer value that is added to the original pixel values 
#Output: returns the adjusted image as an rgb array

        def change_brightness_LH(rgb, value):
            rgb_new = rgb
            rgb_new[:,:int(rgb.shape[1]/2),:] = np.where((255 - rgb) < value,255, rgb+value)[:,:int(rgb.shape[1]/2),:]
            return rgb_new
```

After the manipulation of the brightness of the image, the image is converted to bayer filter encoding before being fed to the network

## (5) Results 

The following section will describe the findings of how the deep learning architecture generalizes to images which differ from the dataset that has been trained for. First the previously shown method to add red, green and blue noise to the image and after that the addition of brightness to the image will be used to see if the output of the net improves the quality again, stays the same or worsens it.

### Zero input validation

<p align="center">
  <img src="/Images/chairs_or_m8.jpg" width="410"/>  <img src="/Images/chairs_rgb0.0m8.jpg" width="410"/>
</p>
<p align = "center">
<b> Fig.N1 - Original image restored with the network of Lamba et al. (left image) and the restored image after being converted by our code without making adjustments to the image (right image). </b>
</p>

Before we can conclude any insights from the results we get it is important to validate that the effect of our changes to the code do not affect the output if no changes have been made to the input. The two images in Figure.N1 show that the original output (left) and the unprocessed output of our added code (right) are exactly the same.

### Addition of Noise

The process of adding noise has been already described above in [SECTION__LINK]. It basically includes first converting the raw image to a RGB image, for which the colour values can be adjusted and then fed back in Bayer format to the neural network.

Three different adjustments that change the red, green and blue values in the same way will be shown, from which it has been found that the net does not generalize very well to them. Other adjustments as different images have been investigated with similar results.

<p align="center">
  <img src="/Images/Chairs_rgb0.1.png">
</p>
<p align = "center">
<b> Fig.N2 - Original unprocessed image opened with raxpy postprocess (left), original unprocessed image opened with our parameters (middle) and the image after adding a noise of variance 0.1 on the red, green and blue channels (right). All images shown are from before being restored by the network. </b>
</p>

Firstly an added noise of 0.1 variance on a standard distribution to the input image has been found undetectable by the human eye, even when zooming in with a multiple of three the input image visually appears completely the same as the original image. 
In Figure.N2 to the left, the unprocessed RAW image can be seen as it would appear when opened with a conventional program. In the middle is the image after being converted to RGB format by our code before any changes have been made to it. Slight differences can be detected between these two rightmost pictures. This comes from the fact that we had to tune the parameters for conversion ourselves for the middle image, whereas the standard programs included in RAWPY perform a little better. The right-most image is showing the RGB image after the noise of 0.1 variance on a standard distribution over the entire colour spectrum has been added. It can be seen that the two rightmost images appear visually exactly the same. 
From this comparison, one might preclude that the neural network would give out exactly the same picture as for an unprocessed picture, but that is unfortunately not true.

<p align="center">
  <img src="/Images/chairs_rgb0.1m8.jpg" width="410"> <img src="/Images/chairs_rgb0.0m8.jpg" width="410">
</p>
<p align = "center">
<b> Fig.N3 - Image with added noise of variance 0.1 on the red, green and blue channels after restoration with the network of Lamba et al. (left image) and the restored image after being converted by our code without making adjustments to the image for comparison (right image). </b>
</p>

It can be seen that in the restored image in Figure.N3 to the left all features can still be seen in their original colour and shape. Yet the restored image is slightly more grainy or pixellated than the input image has been. Comparing it to the same restored image, but then without added RGB noise, the difference in clarity becomes clear.

<p align="center">
  <img src="/Images/Chairs_rgb0.5.png">
</p>
<p align = "center">
<b> Fig.N4 - Original unprocessed image opened with raxpy postprocess (left), original unprocessed image opened with our parameters (middle) and the image after adding a noise of variance 0.5 on the red, green and blue channels (right). All images shown are from before being restored by the network. </b>
</p>

As can be seen above in Figure.N4, increasing the added noise to a 0.5 variance on a standard distribution has been found to be the edge of where the human eye can detect the added noise in the input image. 
The added noise can best be seen in the white area of the window next to the bike. A light hue of red is laid over the originally white area.

<p align="center">
  <img src="/Images/chairs_rgb0.5m8.jpg" width="410">
</p>
<p align = "center">
<b> Fig.N5 - Image with added noise of variance 0.1 on the red, green and blue channels after restoration with the network of Lamba et al. </b>
</p>

When using this as an input image for the network the pixellation becomes now strongly visible in the output as can be seen in Figure.N5 above. 
At a variance of 1 the features within the image start to be obscured by the noise and further increasing the variance leads to the output becoming more and more pixellated until no feature can be recognized anymore at a variance of 3. 

In their published paper 'Restoring Extremely Dark Images in Real Time' M. Lamba and K. Mitra claim that their method 'generalizes better than the state-of-the-art' methods of other researchers without training. 
In our work we have tried and tested different images and noise settings with similar results. This means that for the addition of noise it seems that the claimed good generalization without new training lacks of substance.

### Addition of Brightness

<p align="center">
  <img src="/Images/chairs_rgb0.0m8.jpg" width="410">
</p>
<p align = "center">
<b> Fig.N6 - Image with added noise of variance 0.1 on the red, green and blue channels after restoration with the network of Lamba et al. (left image) and the restored image after being converted by our code without making adjustments to the image for comparison (right image). </b>
</p>


## Discussion
In this reproducibility project two different types of variances (noise and brightness) are added to the input images in order to check the robustness of the model produced by the researchers. The addition of noise showed a very disturbed and unrealistic output image which indicates that the researches model is not well resistant to noise variances. This may be due to the researchers model being trained on high quality images, possibly also having very little differences in quality between the images in that speficic training dataset. We suspect the model to be so well trained to a specific type of image quality that the disturbed results we got are due to the model not being able to handle the added random noise variance. 

The brightness addition showed no real changes in output quality, even a significant addition of brightness to the input image resulted in only a minor difference in the output image compared to a "normal" input image being used. We believe this is because the brightness of images used in the training dataset also differed and therefore the model is able to handle different input brightnesses in a proper way. 

## Conclusion
Returning to the motivation of our research namely testing the model's robustness and checking whether this research actually provides a smart solution or if it's rather a form of parameter tuning and overfitting that has taken place? Looking at our research goals, we found that addition of noise to the input image impacted the output images in a really negative way. Which may imply that the model shows signs of overfitting and parameter tuning and/or that the model is trained using a not well diversified dataset (e.g. only contain the images with the same quality). At the same time this may also be the result of a little diversified training dataset being used by the reseachers. The addition of brightness to the input images returned output images only slightly differing from the output images resulted from the "original" input images. Suggesting that the model recognizes an already bright image and not making that image overly bright. 

In short, the results are really an dubble edged sword with noise addition being process badly by the model but the addition of brightness being handled in a good way. For now we conclude the model not being very robust mostly because of it's bad performance when only little noise is added. Relevant additional research may be done on  verifying the what extend the images from the training dataset are diversified in term of different images qualities (including noisy pictures) and checking wheter the magnitude of brightness also differs within the training dataset. In case the above is not the case, it would be interesting to see how to model's robustness performs when the model is retrained using a well diversified training dataset.


## Sources
[1] Restoring Extremely Dark Images in Real-Time (M. Lamba & K. Mitra, 2021)

