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

Over the years a lot of good methods for low-light image enhancement have been developed. The methods to do so thus actually exist and often consist of some sort of deep learning architecture [SOURCE] but there one main problem with this type of architures namely their high computational costs which on itself leads a slow way of operating these methods. This make most of the existing methods for low-light image enhancement non-practical solutions. In the paper "Restoring Extremely Dark Images in Real-Time" (M. Lamba & K. Mitra, 2021) a fast and memory efficient solution is presented which at the same time produces proper quality (light enhanced) images. 
In this blog post we will test the researcher's sensitivity to variance to the input images. This will be done by applying variances (noise and brightness) to the input images and visually analizing the impact that may have on the quality of the models output images. Our reproduciblity project will thus be focussed on the pre-processing phase. The incentive for this form of reproducibility project is the growing concern in the Deep Learning community when it comes to parameter tuning and overfitting in order to improve results instead of coming up with smart / out of the box improvements. [SOURCE?] Miss ook nog woordje WE weghalen, dat staat vgm in de writing guideline...

(incl. motivation)

## Method

Explination/summary of the method

#### RAW to RGB conversion code snippet

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

#### Adding noise code snippet

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

#### Adding brightness code snippet

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

Text here
(incl. code snippets)

## Results 

The following section will describe the findings of how the deep learning architecture generalizes to images which differ from the dataset that has been trained for. First the previously shown method to add red, green and blue noise to the image and after that the addition of brightness to the image will be used to see if the output of the net improves the quality again, stays the same or worsens it.

### Zero input validation

![After Net Original] (/Images/chairs_or_m8.jpg)   ![After Net RGB 0.0] (/Images/chairs_rgb0.0m8.jpg)

Before we can conclude any insights from the results we get it is important to validate that the effect of our changes to the code do not affect the output if no changes have been made to the input. The two images above show that the original output (left) and the unprocessed output of our added code (right) are exactly the same.

### Addition of Noise

The process of adding noise has been already described above in [SECTION__LINK]. It basically includes first converting the raw image to a RGB image, for which the colour values can be adjusted and then fed back in Bayer format to the neural network.

Three different adjustments that change the red, green and blue values in the same way have been investigated and it has been found that the net does not generalize very well to them.

![Before Net RGB 0.1] (https://github.com/chrisy781/chrisy781.github.io/blob/main/Images/Chairs_rgb0.1.png)
<img src="chrisy781.github.io/Images/Chairs_rgb0.1.png" class="img-responsive" alt=""> </div>
<p align="center">
  <img src="https://github.com/chrisy781/chrisy781.github.io/blob/main/Images/Chairs_rgb0.1.png" height="380">
</p>

Firstly an added noise of 0.1 variance on a standard distribution to the input image has been found undetectable by the human eye, even when zooming in with a multiple of three the input image visually appears completely the same as the original image. 
To the left, the unprocessed RAW image can be seen as it would appear when opened with a conventional program. In the middle is the image after being converted to RGB format by our code before any changes have been made to it. Slight differences can be detected between these two rightmost pictures. This comes from the fact that we had to tune the parameters for conversion ourselves for the middle image, whereas the standard programs included in opencv _OR RAWPY??_ perform a little better. The right-most image is showing the RGB image after the noise of 0.1 variance on a standard distribution over the entire colour spectrum has been added. It can be seen that the two rightmost images appear visually exactly the same. 
From this comparison, one might preclude that the neural network would give out exactly the same picture as for an unprocessed picture, but that is unfortunately not true.

![After Net RGB 0.1] (/Images/chairs_rgb0.1m8.jpg)

It can be seen that in the restored image all features can still be seen in their original colour and shape. Yet the restored image is slightly more grainy or pixellated than the input image has been. Comparing it to the same restored image, but then without added RGB noise, the difference in clarity becomes clear. 

![After Net Unprocessed] (/Images/chairs_rgb0.0m8.jpg)

See above the restored image without added noise.

![Before Net RGB 0.5] (/Images/chairs_rgb0.5.png)

Increasing the added noise to a 0.5 variance on a standard distribution has been found to be the edge of where the human eye can detect the added noise in the input image. 
The added noise can best be seen in the white area of the window next to the bike. A light hue of red is laid over the originally white area.

![After Net RGB 0.5] (/Images/chairs_rgb0.5m8.jpg)

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
