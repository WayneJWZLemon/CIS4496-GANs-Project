Conor Iannetti, Wayne Zheng, Benson Li, Raj Patel

CIS 4496: Projects in Data Science

28 March 2022

## Bi-Monthly Progress Report II

### Introduction:

Within Phase II of our project surrounding the Kaggle competition “I’m Something of a Painter Myself”, we conducted work to improve our GAN model to produce better-looking images that more closely resemble the style of Claude Monet upon visual inspection as well as improve its score on our chosen FID evaluation metric. While the goal of the competition was based on creating a generative adversarial network (GAN) that generates images based solely on the works of Monet, we have also expanded our scope to include the works of other renowned artists. The structure and function of our GAN is maintained since the first phase: two generator models and two discriminator models. A generator is trained to output images in the style of the artist’s painting and another generator is trained to output mundane photographs. For each generator, there is a corresponding discriminator model that takes real images from the target domain and generative images from the generator and predicts whether the input is real or fake. An image is accepted as an output when it is classified by the discriminator as an actual Monet painting when it is generated. Our methods at attempting to improve the performance of our initial model include finding larger datasets of an artist’s paintings, as well as artificially increasing their size by mirroring the images, augmenting the image data within training, and incorporating structural changes to our model found in other GANs that were successful in similar tasks, such as learning rate scheduler. Our end goal of this project is to have several generator models trained by the GAN in the style of various artists accessible by a web portal to which a user could upload a photo and convert it to the style of these artists.

![Phase3HighLevelView](https://user-images.githubusercontent.com/60633000/160512098-86123b1d-7d9c-401e-9a22-ca81665171d1.jpg)
<em>Our Project Objective through CycleGAN</em>

### Progress:

**Data Preparation:**

The Kaggle dataset provides us with just 300 Monet images so we wanted to find a larger dataset and use that to train our model to see if it would result in a better score. We found a dataset gathered by Berkeley with 1193 Monet images in JPG format and 1367 paintings in TFrecord format. Utilizing that dataset resulted in a sizable decrease in our MiFID score as we were able to feed the model more noise which reduced overfitting. Once we had the larger dataset in place, we added other data augmentation techniques during training such as rotations, flips, and random crops to improve the performance of the model. We implemented 270, 180, and 90-degree rotations on the image dataset randomly, and also used horizontal and vertical flipping on some of the images as well as random cropping of image segments. These three techniques are more ways to increase our dataset by giving the model more diverse images to process during training which in turn increases the variance and combats overfitting. Additionally, datasets of paintings by artists other than Monet were gathered from various archival internet sources for art to train the model on other painting styles. Paintings from Van Gogh, Sisley, Picasso, Degas, and Rembrandt were gathered into datasets, with the number of images within them then being increased through the use of horizontal flipping before training occurred. Since these images were not already processed by Kaggle for the competition, certain preprocessing measures that weren’t required for the initial dataset had to be implemented. These include resizing the images to the 256x256 resolution required by the Kaggle competition as well as seeking out and removing the handful of images that were not able to be processed by the model in the 3-channel RGB format. Additionally, the Kaggle competition requires the generated images to be in 3-channel RGB format.

**Methods:**

For this phase, we implemented our model in TensorFlow as well but also had one implementation in the PyTorch framework in order to determine if it was easier to use or allow for any alternative methods that would improve the GAN’s performance in terms of runtime or score. We increased the number of epochs from 30 to 120 as we read that more epochs would train the model longer and give us better results in the quality of images and MiFID score. We stuck with our implementation of the CycleGAN that we described in the first report. To briefly summarize, a CycleGAN trains the model automatically and it also does not need paired examples which is what our project is calling for. The CycleGAN architecture consists of two pairs of generator/discriminator for Domain X. In our case, we have a generator model that is responsible for generating painting-like images from the photo inputs, and another generator model for generating photo-like images from the painting inputs.

<div align="center">Photos → Painting Generator → Paintings</div>

<div align="center">Paintings → Photo Generator → Photos</div>

For each generator, there is also an associated discriminator model that takes the real images from the target domain and the generated images from the generator to predict whether they are real or fake.

<div align="center">Photos → Painting Generator → Generated Paintings → Painting Discriminator → Real/Fake</div>

<div align="center">Paintings → Photo Generator → Generated Photos → Photo Discriminator → Real/Fake</div>

The generator and the discriminator pair work in the adversarial process, the generator learns to better fool the discriminator by creating better images and the discriminator learns to better detect fake images. Altogether, the pair of models will reach an equilibrium as they are under training.
Furthermore, the generator models are not limited to producing images in the target domain, but to translating more reconstructed images in the source domain, this is done by passing the source image into both generator models and it is referred to as a cycle. Cooperatively, the generator models are trained to reproduce the source image and it is referred to as cycle consistency.	

<div align="center">Photos → Painting Generator → Paintings → Photo Generator → Photos</div>

<div align="center">Paintings → Photo Generator → Photos → Painting Generator → Paintings</div>

The CycleGAN architecture encourages cycle consistency by adding a loss function to measure the output image from the second generator in the cycle and the original image. This acts as a regularization of the generator models which guides the image generation process in the target domain towards style translation.

As to our implementation, we have three building blocks, which are the encoder block, transformer block, and decoder block. These blocks made up the generator and the discriminator models. This is a similar design to the original CycleGAN paper.

* The encoder block uses convolutional filters to decrease the image resolution and increase features in the image.

* The transformer block uses convolutional filters to find relevant image features and keep the features constant.

* The decoder block uses convolutional filters to increase the image resolution and decrease the features in the image.

The generator is consisted of all three blocks, first, it uses encoder blocks to enhance features in the source images before passing the data to the transformer blocks, which then extract the important patterns found in the source images, finally, the processed image data is passed back to the decoder to reconstruct the image in the target domain. The discriminator on the other hand is consisted of only the encoder block, this is so-called a PatchGAN model, instead of analyzing the whole image and classifying it as real or fake, this PatchGAN design aims to classify if each N x N patch in an image is real or fake, it will run convolutionally across the image, average all the responses provide a final output for the image.

We also use a constant learning rate scheduler with a linear decay, it allows both the generator and discriminator models to be more stable in later epochs. This technique was included in the original CycleGAN implementation to allow the generators to settle down the generator changes.

**Performance:**

The performance is based on the MiFID (Memorization-informed Frechet Inception Distance) score which is a modification of the FID (Frechet Inception Distance) score. To give a quick recap, the FID score penalizes models that produce images that are too similar to the training set. It uses the inception network to extract features from an intermediate layer and then models the data distributions for the chosen features using mean and covariance. In essence, it tells us how well the model mimics human perception in similarity while promoting diversification in the output. For our first demo, our model produced an 88.48 MiFID score which placed us in 80/85 teams on the Kaggle leaderboard. Using the data augmentations mentioned above and the larger Berkeley dataset, we were able to improve our score to 42.35 for our phase 2 demo. Since Kaggle’s MiFID calculation is based on a comparison to the original dataset of 300 Monet paintings,  we determined that we should have our own method to more accurately track score changes based on our larger Monet or those of other artists. This was able to be achieved by utilizing an FID calculation package built into PyTorch, called pytorch-fid, that can be run through the command line. The scores calculated by this package when comparing the appropriate datasets of original and generated images reflected a similar margin of improvement. Not only did the score go down dramatically, the overall quality of the images greatly improved which is the most important part. The image on the left is an example of the output after phase 1 and the image on the right is the output after phase 2. 

![image](https://user-images.githubusercontent.com/60633000/160512830-94aaee53-e731-4e66-b72d-0b5c9806f2db.png)

The image on the right resembles a painting much more as it does not have many pixelated components compared to the generated image on the left. You can see blotches of different colors in the cloud and sky which is also something you would expect to see in a Monet painting. The vast majority of images indicate a similar increase in quality between the two phases upon visual inspection. The great improvement in our score is therefore backed up by an overall vast improvement in the visual aspects of the generated paintings.

### Plan:

We are satisfied with the progress we have made with this project because of the vast improvement in the MiFID score, FID score, and the overall quality of the images. There is always room to improve, so if we can make some adjustments and get inside the top 20 of the Kaggle competition, we will be even happier. In the next few weeks, we want to see if the model can produce colorblind images in Monet and other artists' styles, by inputting colorblind photos into the model. We also wish to have a user input feature where a user could upload their image and they will be able to see the transformation into the artistic style of their choosing. Finally, we will start working on the interactive dashboard where we hope to showcase the user upload feature as well as some of our favorite images from the dataset transformed into Monet styles and the other artist styles. Additionally, we could also add information about the model, the progress we have made since the beginning of the project, and interesting statistics such as the loss graph or the RGB distribution of a generated image, using a color histogram, in the interactive dashboard.

### Remaining Questions:

The remaining tasks that we need to implement are the colorblind component that we mentioned above and the interactive dashboard. The model produces great Monet images and the images of the other artists are not as good as the Monet ones. If we can come up with a way for the model to work better on other artists then we will explore it. For the colorblind component, we are thinking about adding a package that will convert our output image into what a colorblind person would see. We also need to look into interactive dashboard implementations for data science projects and determine what we can incorporate into ours to create an interesting overview of our model. Additionally, we would need to look into what software, packages, or programming languages are best for creating an interactive dashboard.

### Expected Results: 

In the next few weeks, we hope to incorporate the colorblind filter into our model as well as make progress on the interactive dashboard. The priority of our interactive dashboard is allowing users to upload their own images, so we will first focus on accomplishing that. As far as the model goes, we are not planning on making any drastic changes to the code and structure of our model as we did from Phase 1 to Phase 2 because we have achieved our goal of producing high-quality Monet images. If we are able to make some tweaks and end up inside the top 20 of the Kaggle leaderboard, that would be great but we first want to add the colorblind component and make progress on the interactive dashboard. By the end of phase III, we want to have an interactive dashboard that teaches users about the power of GANs and allows them to convert their personal images into a painting resembling a famous artist. By the end of this project, we hope that everything is running smoothly and others can enjoy our project. 

### References:

* Brownlee, J. (2020, September 1). How to develop a cyclegan for image-to-image			translation with keras. Machine Learning Mastery. Retrieved March 28, 2022,		from https://machinelearningmastery.com/cyclegan-tutorial-with-keras/

* Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2020, August 24). Unpaired image-to-image translation using cycle-consistent adversarial networks. arXiv.org. Retrieved March 28, 2022, from https://arxiv.org/abs/1703.10593 

* https://github.com/vanhuyz/CycleGAN-TensorFlow


