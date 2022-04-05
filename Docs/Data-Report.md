Conor Iannetti, Wayne Zheng, Benson Li, Raj Patel

CIS 4496

4 April 2022

## Artist CycleGAN Data Report
### Data Cleaning and Processing Techniques

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For our model to function, it requires two image datasets: one containing the paintings of an artist, like Monet, and the other being a set of photographs. The CycleGAN model transfers pictures from one domain to another, so a generator that converts a real image from the photo dataset to a Monet-style painting and another generator that does the opposite need to be trained on these images. Then two discriminator models, one for paintings and one for photos, are trained to evaluate the images and attempt to classify if each one is real or generated. Initially, before gathering additional datasets from external sources, the datasets provided by Kaggle for the Monet GAN competition “I’m Something of a Painter Myself” were used. The Kaggle datasets consisted of 300 Monet paintings sized to 256x256 in JPEG and tfrec format and 7028 photos sized 256x256 in JPEG and tfrec format. Since this data was already uniformly sized to 256x256 and otherwise preprocessed for use in the competition, we did not have to implement any function to resize the images or use any substantial preprocessing on this first dataset. After achieving results from a baseline model with this dataset,  we wanted to use a larger dataset of Monet images to improve the model through the inclusion of more training data. We found a dataset gathered by Berkeley which included 1193 Monet paintings in JPEG format and 1367 paintings in tfrec format to achieve this. Since this data had not been preprocessed by Kaggle and ranged in various sizes that were not accepted by our model, we utilized a function to resize all images to 256x256. Once we had all the datasets we needed for Monet, we went to find datasets of other artists’ work so that similar photo-to-painting generators could be trained for their styles. Using sites such as WikiArt, we found datasets containing 1754 Van Gogh images, 1404 Edgar Degas images, 818 Pablo Picasso images, 524 Rembrandt images, and 518 Alfred Sisley images. For those datasets, we again resize all images to 256x256 so they were compatible with our model. The datasets also contained mirrored versions of the paintings to increase the size of the dataset. This technique is a great way to get the most use out of the data we have by doubling the amount of images without having to find additional datasets and it keeps the integrity of the artist’s painting techniques. Additionally, we needed to make sure that the photo dataset and the painting datasets were in 3-channel or RGB channel format and remove any images that did not meet that criteria, as other formats had numbers of channels that were invalid as inputs to our neural network layers. This means that there should be no images that are in formats such as grayscale or HSV inputted into our model. Out of the thousands of images used for training data, only a few had these alternate formats, making it far more simple to remove these during preprocessing rather than alter the model’s structure to accommodate them.

### Feature Extraction Techniques and Data Augmentation Techniques

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In the baseline model constructed during the first phase, we did not implement any data augmentation techniques to add to our data. This, in combination with using the smaller Kaggle dataset, resulted in a poor MiFID score and poor quality of the images upon visual inspection. In our latest model, we implemented random flips, rotations, and crops to give the model more views of the same image to help combat overfitting. More specifically, We utilized horizontal flips, 270, 180, and 90-degree rotations, and random crops of the images in the datasets. Some examples are shown below. 

![image](https://user-images.githubusercontent.com/60633000/161658028-e175a4b8-e91c-4637-9b63-00cded129615.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Using random crops helped the model in being able to generalize the styles better because it will help to combat overfitting. Random flips of the images in the dataset also help the model combat overfitting. Most of the top performing notebooks on the Kaggle leaderboard implemented both random crops and random horizontal flips. Some Kaggle notebooks on the leaderboard chose to randomly increase the brightness and saturation of the painting dataset but we chose not to because we believed that it could compromise the artist’s original style. This is especially important because the colors the painters used at the time are a big reason for why their styles are so distinct from one another. 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For our project, there were not many more feature extraction techniques that we could utilize since our data consists solely of unlabeled images. We did not feel hampered by this though because the data augmentations that we were able to do made massive improvements to our model and overall improved the quality of our images greatly. It is also important to note that many of the high-performing models on the Kaggle leaderboard did little to no exploratory data analysis or categorization of the data.

### Trends in notable features in data
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;With the help of an open notebook on Kaggle, we were able to observe interesting features and trends of the datasets that we have including the color histogram of an individual photo and the pixel value mean of the red, green, and blue channels of the painting dataset and the photo dataset.

![image](https://user-images.githubusercontent.com/60633000/161658223-bcdfed2f-2c91-4944-b363-c3c590ebb4d2.png)

*The image above shows the mean pixel intensity value box plot of the Blue channel of the original Kaggle-provided datasets*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Based on the outputted box-plots of the original Kaggle datasets, we can observe that the Monet paintings dataset had a higher mean pixel intensity value compared to the photo dataset for each of the three channels. From this, we can conclude that the Monet dataset had on average more colors used for each channel than the images in the photo dataset. This is something that makes sense since Monet liked to use a lot of different colors in his paintings for backgrounds. The generators would have to utilize this fact to incorporate more slight variations in the colors to be able to trick the discriminator into believing that it was a Monet painting. This also means that our model’s generated images should have a similar mean pixel intensity value box plot for each of the RGB channels to the Monet dataset. This will serve as a good sanity check for our generated images since if the model is representing the style of an artist well, the box plots should be similar.

### Observed trends that relate to project

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Out of the artist data we have, the model appears to work best on nature or landscape photos instead of more modern objects. This makes sense since these artists were not drawing images of skyscrapers or other modern objects since they did not exist back then. We also did not consider datasets where the images were mostly self-portraits because the model did not work as well on those either. Monet did not make many self-portraits as he was more interested in nature and light for his landscapes. So the model did not have much to go off of when trying to discern or create a portrait in a Monet style. Heading into this project, we were not aware of these facts but it was interesting to learn these as we saw the results and worked through this project. Color histograms are good to use for exploratory data analysis because it gives valuable insight into which colors were used in a painting. This is especially helpful for when we use color histograms for the generated images to see if it matches with the Monet paintings.

![image](https://user-images.githubusercontent.com/60633000/161658389-660828f3-fd29-40c4-838f-17cb936db7c5.png)

*Color Histogram of a Monet Painting*

![image](https://user-images.githubusercontent.com/60633000/161658419-0f407d7b-668d-4074-8e49-311645e5cdf1.png)

*Color Histograms of random images from the Generated Image dataset*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Looking at the color histograms of some random Monet images in the dataset allowed us to see what colors Monet used most frequently. This depends on the image and varies from one to another, but it is still cool to see and is one of the limited techniques we have at our disposal to see some exploratory data analysis for our image dataset. We also have another metric to determine if our generated images resemble the artist’s paintings, which is the box plot for the mean pixel intensity value for the red, blue, and green channels. If we observe that the box plot for our generated images is similar to the box plot of the paintings dataset, we can say that our generated images have done a good job at mimicking the artist’s unique painting style.

![image](https://user-images.githubusercontent.com/60633000/161658465-60fc20bf-ee8a-451b-9c40-5f4a293c2041.png)

*The image above shows the mean pixel intensity value box plot of the red channel of the original Kaggle datasets, Generated Images, and the Berkeley dataset*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We can also look at the box-plots of the mean pixel intensity value of each image datasets, to determine if our model is performing well. Based on the outputted box-plots of the original Kaggle datasets, Generated Images, and the Berkeley dataset, we can observe that the generated images have a similar box-plot distribution, for all three channels, to both of the painting datasets. This is a good sign for our generated images because it indicates that our model is mirroring the style of Monet relatively well because it is using a similar amount of colors when compared to the paintings dataset.

![image](https://user-images.githubusercontent.com/60633000/161658618-49608868-40f8-4938-a606-8539dbecd7a3.png)
![image](https://user-images.githubusercontent.com/60633000/161658640-26cade7f-d411-44a9-ab1f-ac0691353036.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For our project we wanted to attempt some categorization or separation of the dataset, so we wanted to test if it was possible to organize the dataset based on light-colors or dark-colors used in the paintings. We hoped that this would provide some useful feedback on how our model would deal with certain types of images, so that we can adjust our model accordingly. To separate the image dataset based on light-colors and dark-colors, we ran the k-means clustering algorithm, with k=2, on both datasets to see if it could categorize it well enough. Looking at the original Kaggle provided dataset, the amount of images in the two categories were about roughly the same. However, for the Berkeley dataset, which contains more Monet paintings, there were about a hundred more light-colored paintings compared to the dark-colored paintings. In other words, around 54 percent of the dataset were paintings with light colors and around 46 percent of the dataset were paintings with dark colors. This almost even split between the light and dark colors helped us understand the data better by gauging whether or not the model performed better or worse on light colored paintings or dark colored paintings. If the dataset consisted of mostly lighter colored images, then we would expect to perform better on images that were more lightly colored than darkly colored and vice versa. 

![image](https://user-images.githubusercontent.com/60633000/161658711-ae2ef611-1760-4471-94c9-38aa1c4e3af5.png)
![image](https://user-images.githubusercontent.com/60633000/161658730-2398deb0-be32-4dd2-8138-b2585a6e8368.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Another way we were able to garner more insight about the various styles our painters used was looking at a color histogram of the same photo transformed into the respective styles of our artists. We can see that Sisley used the most blue to transform the image followed by Monet, Rembrandt, Degas, and Picasso. Red was not used much by any of the artists for this specific image since it involves trees and the blue sky coming through them. We can observe that Sisley also used the most green out of the artists as his image involves green leaves whereas the other artists chose to incorporate some red and yellow into the leaves. Overall, even though we are looking at only one image, we can expect to see some of the same stylistic components involved with other images in our dataset, especially those revolving around natural views. 

![image](https://user-images.githubusercontent.com/60633000/161658778-441ddc07-d093-4158-b95f-44bffa602c70.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Lastly, we used the OpenCV library with the provided CascadeClassifier to specifically target human objects in the Berkeley dataset, we ran the detection algorithm on image features such as “full-body,” “eye,” “upper body,” and “lower body,” where the algorithm would spot out the specific images and put green boxes surrounding the detected human features. Then we manually looked through the spotted images because the algorithm is not completely accurate, for example, it sometimes identified ships to be humans and falsely claimed landscape images had humans in them. To avoid misinformation, we inspected the spotted images ourselves and delete all duplicated images. Finally, we found 24 images from the 1193 images that have humans in the painting using the detection algorithm.

### Does the data foretell any issues that may arise in later stages of project lifestyle?

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The data and results we got from it tell us that it works better on some artists' styles based on others. For example, out of the handful of different artists that we included, Monet's images were always the highest quality images. That can be explained maybe by the data that we were using to train the model that was more suited for Monet’s style than others. Monet liked to paint natural landscapes, so there are not many paintings of people or objects in general. Whereas an artist like Picasso loved to paint self-portraits so the model does not have many examples to learn from since it is not used for discerning portraits or non-natural landscape images. We attempted to solve this problem by carefully picking the datasets that would perform well. We avoided using a Picasso dataset that consisted mainly of self-portraits and did the same for the other artists. 

![image](https://user-images.githubusercontent.com/60633000/161658839-b5496cd3-30b6-4521-8741-8865354100f8.png)

*Example of a photo containing a person not being converted to the Monet style well*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The above photo shows how our model does not perform well when the input photo contains a person, as shown by the black splotch in the middle of the generated image. Monet was typically focused on drawing landscapes and other natural settings. This is an issue that will be difficult to resolve since Monet typically did not paint people, which makes it incredibly difficult for the model to adapt.

![image](https://user-images.githubusercontent.com/60633000/161658887-15313b09-efed-4dc4-b178-818ffb69bc1b.png)

*Example of a photo containing modern architecture not being converted to the Monet style well*

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This image shows another issue with our model, specifically how it sometimes does not perform well on pictures of modern landscapes or architecture. This can also be attributed to a lack of modern architecture paintings in the paintings dataset. This is an issue that will be difficult to resolve since Monet did not paint modern architecture, making it hard for the model to adapt to the inputted images of this type.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Another issue that arose with our project related to data with the size of the datasets we found for the other artists. The model performed much better with larger datasets as expected. So if we had a dataset with 500 images of one artist and only a 200 image dataset of another artist, it would not produce high-quality images of the artist from the smaller dataset when compared to the generated images for the 500 images dataset. Thus, we had to be careful when choosing the datasets of the other artists so the generated images would not be greatly negatively affected by the size. Our datasets for Rembrandt and Sisley are much smaller than the datasets for Van Gogh and Monet so that could be a reason why the model did not perform as well on those artists as the other larger painting datasets.





