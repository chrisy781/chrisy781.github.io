# Group 11 Reproduction Extremely Dark Images Project
On this page we will give an overview of our reproduction efforts of the orginal research paper called "Restoring Extremely Dark Images in Real Time".

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

## Introduction

Over the years a lot of good methods for low-light image enhancement have been developed. The methods to do so thus actually exist and often consist of some sort of deep learning architecture [SOURCE] but there's one main problem with this type of architures namely their high computational costs which on itself leads a slow way of operating these methods. This makes most of the existing methods for low-light image enhancement non-practical solutions. In the paper "Restoring Extremely Dark Images in Real-Time" (M. Lamba & K. Mitra, 2021) a fast and memory efficient solution is presented which at the same time produces proper quality (light enhanced) images. 
In this blog post we will test the sensitivity of the researcher's model to variances applied to the input images. In other words how robust is the model created by reseachers? 
This will be done by applying variances (noise and brightness) to the input images and visually analizing the impact this may have on the quality of the model's output images. The reason for choosing both noise and brightness as variances to the input image is because of the following: first of all, both methods of variance allow us to check the model's sensitivity in an easy to control manner, by gradually increasing these variances and analyze it's influence on these output images. Secondly, noise is chosen as a variance because this addition of a random distruption to the input image allows us to analyse how well the model can handle this type of random variance. Lastly, the adjsutment of brightness is chosen because it allows us to see how the researcher's model performs when the (dark) input image is brightened up. Our reproduciblity project will thus be focussed on the pre-processing phase. The incentive for this form of reproducibility project is the growing concern in the Deep Learning community when it comes to parameter tuning and overfitting in order to improve results instead of coming up with smart / out of the box improvements. [SOURCE?] 
Miss ook nog woordje WE weghalen, dat staat vgm in de writing guideline...

(incl. motivation)

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

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

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

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for
```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

#### (C) RGB to RAW conversion code

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for
```markdown
- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

Text here
(incl. code snippets)

## (5) Results 

The following section will describe the findings of how the deep learning architecture generalizes to images which differ from the dataset that has been trained for. First the previously shown method to add red, green and blue noise to the image and after that the addition of brightness to the image will be used to see if the output of the net improves the quality again, stays the same or worsens it.

### Zero input validation

<p align="center">
  <img src="/Images/chairs_or_m8.jpg" width="350"/>  <img src="/Images/chairs_rgb0.0m8.jpg" width="350"/>
</p>
<p align = "center">
<b> Fig.N1 - Original image restored with the network of Lamba et al. (left image) and the restored image after being converted by our code without making adjustments to the image (right image). </b>
</p>

Before we can conclude any insights from the results we get it is important to validate that the effect of our changes to the code do not affect the output if no changes have been made to the input. The two images in Figure.N1 show that the original output (left) and the unprocessed output of our added code (right) are exactly the same.

### Addition of Noise

The process of adding noise has been already described above in [SECTION__LINK]. It basically includes first converting the raw image to a RGB image, for which the colour values can be adjusted and then fed back in Bayer format to the neural network.

Three different adjustments that change the red, green and blue values in the same way will be shown, from which it has been found that the net does not generalize very well to them. Other adjustments as different images have been investigated with similar results.

<p align="center">
  <img src="/Images/Chairs_rgb0.1.png" height="500">
</p>
<p align = "center">
<b> Fig.N2 - Original unprocessed image opened with raxpy postprocess (left), original unprocessed image opened with our parameters (middle and the image after adding noise or 0.1 on red, green and blue channels (right). All images shown are from before being restored by the network) . </b>
</p>

Firstly an added noise of 0.1 variance on a standard distribution to the input image has been found undetectable by the human eye, even when zooming in with a multiple of three the input image visually appears completely the same as the original image. 
To the left, the unprocessed RAW image can be seen as it would appear when opened with a conventional program. In the middle is the image after being converted to RGB format by our code before any changes have been made to it. Slight differences can be detected between these two rightmost pictures. This comes from the fact that we had to tune the parameters for conversion ourselves for the middle image, whereas the standard programs included in opencv _OR RAWPY??_ perform a little better. The right-most image is showing the RGB image after the noise of 0.1 variance on a standard distribution over the entire colour spectrum has been added. It can be seen that the two rightmost images appear visually exactly the same. 
From this comparison, one might preclude that the neural network would give out exactly the same picture as for an unprocessed picture, but that is unfortunately not true.

<p align="center">
  <img src="/Images/chairs_rgb0.1m8.jpg" height="380">
</p>

It can be seen that in the restored image all features can still be seen in their original colour and shape. Yet the restored image is slightly more grainy or pixellated than the input image has been. Comparing it to the same restored image, but then without added RGB noise, the difference in clarity becomes clear. 

<p align="center">
  <img src="/Images/chairs_rgb0.0m8.jpg" height="380">
</p>

See above the restored image without added noise.

<p align="center">
  <img src="/Images/Chairs_rgb0.5.png" height="380">
</p>

Increasing the added noise to a 0.5 variance on a standard distribution has been found to be the edge of where the human eye can detect the added noise in the input image. 
The added noise can best be seen in the white area of the window next to the bike. A light hue of red is laid over the originally white area.

<p align="center">
  <img src="/Images/chairs_rgb0.5m8.jpg" height="380">
</p>

When using this as an input image for the network the pixellation becomes now strongly visible in the output as can be seen in above image. 
At a variance of 1 the features within the image start to be obscured by the noise and further increasing the variance leads to the output becoming more and more pixellated until no feature can be recognized anymore at a variance of 3. 

In their published paper 'Restoring Extremely Dark Images in Real Time' M. Lamba and K. Mitra claim that their method 'generalizes better than the state-of-the-art' methods of other researchers without training. 
In our work we have tried and tested different images and noise settings with similar results. This means that for the addition of noise it seems that the claimed good generalization without new training lacks of substance.

### Addition of Brightness

## Discussion

Text here

## Conclusion

Text here
(what did we see and what did we expect)

## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/chrisy781/chrisy781.github.io/edit/main/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/chrisy781/chrisy781.github.io/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
