# Deep-Learning-Model-for-Artistic-Style-Transfer
**INTRODUCTION**  

Given a content image and a style image , we try to genereate an image that follows the content of the former and style of the latter.
We’ll take the base input image, a content image that we want to match, and the style image that we want to match. We’ll transform the base input image by minimizing the content and style distances (losses) with backpropagation, creating an image that matches the content of the content image and the style of the style image.
We are using the CNN based network architecture VGG19, an image classification network, pretrained on ImageNet dataset. These intermediate layers are necessary to define the representation of content and style from our images. For an input image, we will try to match the corresponding style and content target representations at these intermediate layers.
For calculating the content loss, we pass the network both the desired content image and our base input image. This will return the intermediate layer outputs from our model. Then we simply take the euclidean distance between the two intermediate representations of those images. For calculating style loss, we compare the Gram matrices of the two outputs. We then compute the gradient loss for the loss function.

**SET UP**  
1. The python notebook contains the code to solve the problem. The version is python 3.  
2. Dependencies to be installed: tensorflow<2.11  
3. The notebook is to be run after importing the required images from your local directory.Here we used 1 content image(night scene) and 3 style images(dali style, manga style, van gogh style) that was imported from personal directory.  
4. The location of the images are given in cell block number 10 as follows:
   img_dir = path to the directory where the images are stored,  
   content_path = path to the content image,  
   style_paths = a list of paths to the three style images.

**PERFORMANCE**  
The content loss gives the content accuracy and style loss gives the style accuracy, which is essentially how different the current image is in terms of content and style from the content and style image respectively. For example, these are the loss function scores at every 100th iteration for one of the style image. There are 1000 iterations and we can see that that the content loss, style loss and total loss decreases from power of 8 to a power of 6.  
Iteration: 0  
Total loss: 4.8348e+08, style loss: 4.8348e+08, content loss: 0.0000e+00, time: 0.0609s  
Iteration: 100  
Total loss: 3.2357e+08, style loss: 3.1284e+08, content loss: 0.1073e+00, time: 0.0612s  
Iteration: 200  
Total loss: 2.1958e+06, style loss: 1.1581e+06, content loss: 1.0377e+06, time: 0.0779s  
Iteration: 300  
Total loss: 1.8699e+06, style loss: 9.6339e+05, content loss: 9.0651e+05, time: 0.0625s  
Iteration: 400  
Total loss: 1.5383e+06, style loss: 7.3732e+05, content loss: 8.0098e+05, time: 0.0669s  
Iteration: 500  
Total loss: 1.4000e+06, style loss: 6.5838e+05, content loss: 7.4164e+05, time: 0.0625s  
Iteration: 600  
Total loss: 1.2907e+06, style loss: 5.9327e+05, content loss: 6.9742e+05, time: 0.0643s  
Iteration: 700  
Total loss: 1.2350e+06, style loss: 5.6432e+05, content loss: 6.7065e+05, time: 0.0781s  
Iteration: 800  
Total loss: 1.2240e+06, style loss: 5.8301e+05, content loss: 6.4101e+05, time: 0.0625s  

**LIMITATIONS AND POTENTIAL IMPROVEMENTS**  

1.VGG is a deep network with a significant number of parameters, which makes it computationally expensive and memory-intensive for image transfer tasks.  
2.Since VGG was specifically designed for image classification, it is not necessary that the layers that we are extracting has features that is most suited for an image transfer process, hence we need to use not only convolutional layers but also deconvolutional layers to reconstruct the image.  
3. A total variation regularization term may be added to encourage smoothness.
