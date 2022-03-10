# CIS4496-GANs-Project Charter
## Contributors
[Conor P Iannetti](https://github.com/ConorIannetti), [Benson Li](https://github.com/GitHubFunTime), [Raj Patel](https://github.com/RAJPATEL05), [Wayne Zheng](https://github.com/WayneJWZLemon)
## Problem description
<figure>
  <img
  src= "https://user-images.githubusercontent.com/60633000/151446351-7de2fbb1-f2c0-4cee-8f3d-06d19468d25f.png"
  alt="The beautiful MDN logo.">
	<figcaption><em>Figure 1. Project Objective High-Level View</em></figcaption>
</figure>
	  
This project is all about utilizing generative adversarial networks (GANs) to attempt to imitate the works of past renowned artists, specifically Claude Monet. By making use of computer vision, GANs can mimic objects in a very convincing way and for this problem, the goal is to trick the classifier into thinking we have created an actual Monet style painting. The use of GANs is an important and exciting field of study in data science. It delivers promising models with the ability to generate realistic examples across a wide range of problem domains, in this case, it is translating photos to paintings. There are multiple benefits of utilizing GANs, including the extension to reinforcement learning, a new way to generate more data instead of data augmentation, and many other usages. One notable GAN application is producing new data in drugs dataset for existing illnesses, which allows researchers to invent new possible treatments. Our project aims to apply GANs to create a program in which the users can experience the potential of making artwork with machine learning techniques. Art is important in fostering creativity, curiosity and opening up one’s imagination. Computer vision and GANs will help users experience the stylistic components of art from centuries ago right on their screen and they can tinker with the GANs to create their own. 

## Project Scope
The data science solution we are trying to develop is a GAN that develops anywhere between 7,000 to 10,000 Monet-style images. We will have to create a generator model which is a neural network that will create the images. Once we have developed that, we will have to test it on the discriminator, whose job is to accurately classify the real vs generated images from the generator model. We will use the popular standard metric for GANs evaluation, specifically, the FID score (a measure of similarity between two datasets of images) and MiFID (a modified FID score to penalize models producing images too similar to the training set) to evaluate the performance of our model. However, it is very tricky to train GANs as problems like overfitting, vanishing gradients, and the curse of dimensionality can cause the model to degrade at one point. The model may generate images with excellent evaluation scores, but look completely unrealistic, thus, we will also manually inspect the generated images after the training. We will deliver a highly functional generator model that will generate 7,000 to 10,000 Monet-style images as accurately as possible to achieve a high success rate. The user will be able to hopefully upload their own images that they want to be replicated in a Monet style or even change some parameters to achieve a unique image if they wished. We will also likely expand the features of projects to include not just Monet style, but also other artists and that can lead us to find more datasets and potentially web crawl images using Scrapy on sites like Pinterest and DeviantArt. Lastly, we may need to incorporate the natural language processing aspect to the project, in that case, we would want to add in a task where users can put in a text string and our model will generate a painting in the desired style that the user chooses at runtime. This text-to-image translation would require some additional work. First, we would need to have a dataset such that there are text descriptions for every painting in the dataset, in which we would likely need to perform human labor works of adding text labels if there are no such datasets available online. Then, we would need to modify the generator and the discriminator accordingly, specifically, the generator would need to accept a 256-dimensional text embedding so that the generated images will be aligned with its input description. The discriminator on the other hand would need to perform two predictions: whether the generated image is real or fake, and the likelihood of the given image and its text aligned with each other. However, this text translation task is secondary to the main task of developing GANs for the styles of various artists and may not be achieved within our current timeframe.
	
## Metrics
Generative adversarial networks (GANs) are a type of deep-learning generative model that has been remarkably effective for creating high-quality and large synthetic images. However, since the whole generative approach involves the generator and the discriminator models to train each other, there is no objective measure for the generator model, which makes it difficult to compare the performance of various models.

This leads to a situation where multiple measures were introduced by the data science community, and there is not a standard and consensus as to how you should evaluate the models.
1. Manual GAN Generator Evaluation
	  
	  * Many GANs practitioners will first begin their evaluation process via the manual assessment of images, where people would compare the quality and diversity of the synthesized images in relation to the target domain. This method is one of the most common and intuitive ways to evaluate GANs, however, it is subjective and often includes biases of the reviewer who needs to have prior knowledge about the target domain. However, their knowledge is limited by the number of images that can be reviewed in a short period of time.

2. Automated GAN Generator Evaluation
	  
	  * A popular approach for summarizing the generator performance is to use the distance measure, such as Euclidean distance between the image pixel data, and it is used to select the most similar generated images. The nearest neighbor can be used to give insight into how realistic the synthesized images happen to be.
	  	  
	  * The Inception Score is an objective metric for evaluating the performance of the generative models. It was initially proposed by Tim Salimans, et al, in their respected 2016 paper named “[Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498).” It was an attempt to remove the human interaction in the evaluation process, the score involves utilizing a pre-trained deep learning neural network model for image classification to classify the generated images. (The inception v3 model described by Christian Szegedy, et al. in their 2015 paper titled “[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567).” From the score, one can capture two important properties about the synthesized images, which are the image quality and the image diversity.
	  
	  * Frechet Inception Distance, or FID, is a metric for evaluating the quality of generated images and was developed to target the evaluation of GANs. FID along with Inception Score are both commonly used in recent publications as the standard for evaluation methods of GANs. FID does not capture how generated images compare to real images, it only evaluates the statistics of a group of generated images to the statistics of the real images. It uses the v3 model like the Inception Score. The coding layer of the model would produce feature maps of the input image. Then these activations are calculated for a collection of real and generated images. A lower FID score indicates the generative images’ statistical properties are closer to the real images, hence more realistic. (*The FID score is also the evaluation metric used by Kaggle in this competition)
	  
	  * Memorization-informed FID is a modified FID score used to penalize models producing images too similar to the training set. Lower memorization distance is associated with more severe training sample memorization. (* MiFID is the major metric used by Kaggle in this competition)

## Architecture
This project consists of largely unsupervised learning tasks that create an exceptional generative model that can produce high-quality Monet-style paintings from random input photos. We will likely use a cloud-based Jupyter Notebook like the Kaggle Notebook or the Colab Notebook to implement code in a team setting where we can individually see code changes made by others and collaboratively write code at the same time. We will make use of Kaggle’s TPU acceleration to train the models efficiently when GPU resources are limited. After we decide that our implementation is sufficient, we will convert the Jupyter Notebook files to Python scripts to take advantage of the computing power. We are looking to implement different generator models and different discriminator models using PyTorch and TensorFlow. If time permits, we would like to deploy our model to a web interface using Django or Flask with hosting services like AWS to allow users to have the access to upload their own image and see the resulting synthesized painting generated by our model. If not, we would at least create a web-based interactive dashboard using libraries like Plotly and Matplotlib where users can view different visualizations of data about our project. 
	  
## Plan
| Date | Description | Goals |
| --- | --- | --- |
| 2-14-2022 | First Project Demo | To have a baseline model developed after having already explored existing solutions |
| 3-21-2022 | Second Project Demo | Work on improving our model to optimize the solution. We would also like to compare our model with other possible models |
| 4-18-2022 | Third Project Demo | Improve the Phase 2 model and test the model against new data to make sure it behaves as it should and tweak if necessary |
| 4-25-2022 | Interactive Dashboard Demo | To have created an interactive dashboard for our project. We will look at using services like Django to deploy it |
	 
## Personnel
1. Conor
* Analyzing existing solutions 
* Working on progress reports
* Collecting dataset and Analyzing dataset
* Implement MiFID
	  
2. Benson
* Focuses on Data report
* Analyzing an existing solution
* Model training using TensorFlow
* Working on interactive dashboard

3. Raj
* Analyzing existing solutions
* Working on progress reports
* Assisting with interactive dashboard 
* Bi-monthly Progress Report

4. Wayne
* Analyzing one existing solution to GANs related task
* Implementing the generator and the discriminator using PyTorch
* Working on the project charter revision
* Keep track of the project management using GitHub/Trello

## Communication
* The team will make extensive use of GitHub for code management and version control, we will have a repository containing our Python code and important documentation as markdown files. GitHub in itself has many features that are extremely helpful for team projects like the project board that allows us to organize and prioritize our work, and the issues tag that helps us track ideas, feedback, or bugs that occur in the current development of the project. As this project is a semester-long project, we will likely be developing different components in various stages, and when there are multiple people working on the same project, it’s important to track changes about who changed what, what new files are being created, and such. GitHub will take care of these changes and if something breaks the whole project, the team can simply revert the specific change that causes the issue to get back on the workflow if needed.
* We are going to use Discord for regular communication within the team outside of class time. This is a relatively big project, so it’s important to have everyone on the same page, thus we need to stay in touch and have conversations about how everyone is doing for their assigned tasks. If someone has a question, they can reach out to other team members and we can work out a solution together. We will make a private server on Discord where all team members can chat about their thoughts and give feedback on the project.
* We will use the project management tool on GitHub for planning tasks on a weekly basis and assigning tasks to different team members. We have four people in a team, thus we should assign everyone to different tasks to maximize efficiency. We are going to create a Github project board that splits tasks into different categories like ideas, to-do, doing, and done. It allows us to organize the tasks and see what we need to do for the project in a clear and vivid fashion rather than reading a long list of bullet points which can be cumbersome.

