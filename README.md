# Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes
[image of problem]
![](https://i.imgur.com/PMdmW2Z.gif)

[almost done] bedeutet, dass Quellen und Bilder fehlen/noch nicht korrekt sind

# Introduction [almost done]
Obtaining the 3D shape of an object from a set of images is a well studied problem. The corresponding research field is called Multi-view 3D-reconstruction. Many proposed techniques achieve impressive results, but fail to reconstruct transparent objects. Image based transparent shape reconstruction is an ill-posed problem. Reflection and refraction lead to complex light paths and small changes in shape might lead to completely different appearance. Different solutions to this problem have been proposed, but the acquisition setup is often tedious and requires a complicated setup. In 2020 a group of researchers from the University of California in San Diego state they have found a technique that enables the reconstruction of transparent objects using only a few unconstrained images taken with a smartphone. This blog post will provide an in-depth look into the paper “Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes” by Zhengqin Li, Yu-Ying Yeh and Manmohan Chandraker.



# Previous work [almost done]
Transparent object reconstruction has been studied for more than 30 years [17]. A full history of related research is out of scope. Approaches were typically based on physics, but recently also deep learning researchers have attempted to find a solution. "Through the looking glass" is able to solve this challenging problem by combining the best of both worlds. The following is a brief overview of important work of the last years. A common characteristic of both approaches is the use of synthetic images.

## Physics-based approaches [almost done]
The foundation of most research is the vast array of knowledge in the field of optics. The authors use this knowledge to formulate their mathematical models and improve them utilizing one manifestation of optimization algorithms. Recent research [3,4] uses synthetic datasets to overcome the hurdle of acquiring enough training data. This comes at the cost of a mismatch between the performance on real world data compared to sythetic datasets. When testing on real world data, these approaches typically require a complicated setup including multiple cameras taking images from various fixed viewpoints.
A paper from 2018 by Wu et al. [4] captures 22 images of the transparent object in front of predefined background patterns. The camera viewpoints are fixed, but the transparent object rotates on a turntable. 

The underlying optics concepts used in physics-based papers are fundamental to understand "Through the looking glass". The section [Overview of optics fundamentals](#optics-fundamentals) will introduce these topics.

## Deep Learning-based approaches [almost done]
The setup of deep learning based approaches is usually simpler. Using RGB [5]/RGB-D[6] images of the transparent object, models learn to predict e.g. the segmentation mask, the depth map and surface normals. These models are typically based on Encoder-Decoder CNNs. Deep learning methods inherently need far more data and therefore also leverage synthetic datasets. 



# Important concepts [work in progress]
Prior to introducing the proposed methods some basic concepts from the fields of optics and 3D graphics shall be clarified. 
## Normals and color mapping [almost done]
A normal is a vector that is perpendicular to some object. A surface normal is a vector that is perpendicular to a surface and it can be represented like any other vector $[X,Y,Z]$.

A normal map encodes the surface normal for each point in an image. The color at a certain point indicates the direction of the surface at this particular point. The color at each point is described by the three color channels $[R,G,B]$. A normal map simply maps the $[X,Y,Z]$-direction of the surface normals to the color channels.

[Show an image that maps the x,y,z coordinates of the surface normals to the colors, This image of a mouse is represented by this normal map.]

## Optics fundamentals [work in progress]
Light propagates on straight paths through  vacuum, air and homogeneous transparent materials. At the interface of two optically different materials things get complicated. Several optical effects change the angle of propagation. In most configurations a single path is split into two paths. In rare cases total reflection with only one emanating light path occurs. Optics found ways to predict the behavior of light. Snell's law of refraction and the Fresnel equations allow to calculate precise angles of reflection and refraction and the relative energy proportions. The index of refraction (IOR) is the most important optical material property. The different optical effects at inferaces change light path angles and splits as a function of light polarization, incidence angle, surface normal and IOR difference. More about basic optical concepts can be found in  [13] and [14]. The Fresnel equations are more complex as they account for polarization what is normally ignored in computer graphics. You can read about them at [15]. In simple terms the Fresnel equations help us to calculate the fraction of light that gets reflected and the fraction that gets refracted.



# Proposed method [work in progress]
## Problem setup [work in progress]
2-bounce assumption


What do N^1 and N^2 exactly mean? What does one pixel in N^1/N^2 mean?

As input, the model requires a small number of unconstrained images of the transparent object and the corresponding segmentation masks for all images. It also needs the environment map and the index of refraction of the material of the transparent object.
The approach usually returns good results with just 10 input images, but some more complex shapes might need 12 input images.

## Overview [almost done]
The authors propose the following contributions:
* A physics-based network that estimates the front and back normals for a single viewpoint. It leverages a fully differentiable, also physics based, rendering layer.
* Physics-based point cloud reconstruction using the predicted normals. 
* A publicly available synthetic dataset [2a] containing 3600 transparent objects. Each object is captured in 35 images from random viewpoints, resulting in a total of 120,000 high-quality images.

The model starts off by initializing a point cloud of the transparent shape. This point cloud is inaccurate, but serves as a good starting point for further optimization. In the next step the physics-based neural network estimates the normal maps $N^1$ and $N^2$ for each viewpoint. The predicted, viewpoint-specific features, will then be mapped onto the point cloud. Finally, the authors use point cloud reconstruction to recover the full geometry of the transparent shape. The model is trained using the synthetic dataset and can be tested on real world data. The code is publicly available on Github [2].

## Space carving [almost done]
Given a set of segmentation masks and their corresponding viewpoint, the space carving algorithm, first introduced by Kutulakos et al. in 1998 [11 $\leftarrow$ improved version from 2000] is able to reconstruct a good estimate of the ground truth shape, called visual hull (more info here: [paper, 11], [video, 12]). Using the visual hull, the front and back normals can now be calculated. They will later be referred to by the notation $\tilde{N^1}$, $\tilde{N^2}$, or by the term visual hull initialized normals. These normals already provide a good estimate for the ground truth normals.

## Normal prediction [almost done]

### Differentiable rendering layer [almost done]
The model utilizes differentiable rendering to produce high quality images of transparent objects from a given viewpoint. The renderer is physics based and uses the Fresnel equations and Snell's law to calculate complex light paths. Differentiable rendering is an exciting, new field that emerged in the last years. This video [16] provides a good introductory overview of the topic.
To render the image from one viewpoint, the differentiable rendering layer requires the environment map $E$ and the estimated normals $N^1$, $N^2$ for this particular viewpoint. It outputs the rendered image, a binary mask indicating points where total internal reflection occurred and the pointwise rendering error with masked out environment. The rendering error map is calculated by comparing the rendered image to the ground truth image.
[image of the output of the differentiable rendering layer]

### Cost volume [almost done]
The search space to find the correct normal maps $N^1$, $N^2$ is enormous. Each point in the front and back normal map could have a completely different surface normal. As stated before, the visual hull initialized normal maps are a good estimate for the ground truth normal maps. Therefore, the search space will be restricted to normal maps close to the visual hull initialized normals. To further reduce the search space, $K$ normal maps for the front and back surface are randomly sampled around the visual hull initialized normal maps. K normal maps for both $N^1$ and $N^2$ lead to $K \times K$ combinations. According to the authors, $K=4$ gives good results. Higher K-values are only increasing the computational complexity, but not the quality of the normal estimations. The entire cost volume consists of the front and back normals, the rendering error and the total internal reflection mask:
[image of the cost volume]


### Normal prediction network [almost done]
To estimate the surface normals an encoder-decoder CNN is used. The cost volume is still too karge to be fed into the network. Therefore, the authors first use learnable pooling to perform feature extraction on the cost volume. They concatenate the condensed cost volume together with
* the image of the transparent object
* the image with masked out environment
* the visual hull initialized normals
* the total internal reflection mask
* the rendering error

and feed everything to the encoder - decoder CNN. [insert loss function here] is used as the loss function. It is simply the L2 distance between the estimated normals and the ground truth normals.

## Point cloud reconstruction [almost done]
The normal prediction network gives important information about the transparent object from different viewing angles. But somehow the features in the different views have to be matched with the points in the point cloud. Subsequently, a modified PointNet++ [Quelle!!]  will predict the final point locations and final normal vectors for each point in the point cloud. Finally the 3D point cloud will be transformed into a mesh by applying poisson surface reconstruction. [
Michael Kazhdan, M. Bolitho, and Hugues Hoppe. Poisson Surface Reconstruction. In Symp. on Geometry Processing, pages 61–70, 2006.] 


### Feature mapping [almost done]
The goal of feature mapping is to assign features to each point in the initial point cloud. In particular, these features are the normal at that point, the rendering error and the total internal reflection mask. The

[ich möchte hier noch erklären, dass es sehr einfach ist von einem 3D-Punkt des shapes den 2D Punkt in den Bilder/normal maps herauszufinden, weil wir die Viewpoints der Bilder kennen]
Map point in 3D space onto each image to figure out the value at this point.

In some cases, a point might not be visible from a particular angle, if so, it will not be taken into account during feature mapping. For each point there are usually 10 different views and sets of features. It now has to be decided, which views to take into account when creating the feature vectors. The authors try three different feature mapping approaches:

#### Rendering error based view selection [work in progress]
Select the view with the lowest rendering error
#### Nearest view [work in progress]
Select the nearest view
#### Average fusion [work in progress]
Average the information over all views

### Modified PointNet++ [almost done]
Given the mapped features, PointNet++ performs point cloud reconstruction. It predicts the final point cloud and the corresponding normals. The authors were able to improve the predictions by modifying the PointNet++ architecture. In particular, they replaced max-pooling with average pooling, passed the front and back normals to all skip connections and applied feature augmentation. See [evaluation] for a quantitative comparison between the out-of the box PointNet++ architecture and the modified version.

To get the best results, the authors try three different loss functions:
#### Nearest view based loss [work in progress]

#### View dependent loss [work in progress]

#### Chamfer distance loss [work in progress]
The chamfer distance is a distance measure between 2 point clouds
 
### Poisson surface reconstruction [almost done]
Poisson surface reconstruction was first introduced in 2006 [quelle!] and is still used in recent papers. It takes a point cloud and surface normal estimations for each point of the point cloud and reconstructs the 3D mesh of the object.



# Evaluation [work in progress]
## Qualitative results [work in progress]
While comparing the ground truth to the reconstructed objects, we can see that the quality of the reconstructions are already really good. At first sight, the scenes look reasonable and no big differences between ground truth and reconstructions are visible.

## Quantitative results [work in progress]
As described earlier, the authors try three different loss functions and three different view-selection strategies. The best combination of loss function and view selection is the rendering error-based view selection with a chamfer distance loss. It achieves the best results across all metrics and greatly improves the visual hull initialized reconstruction (vh10).


In Table 3 the authors examine the effect of parts of their approach using different metrics. The first column shows the visual hull initialized reconstruction, the second column shows that bare minimum of the model. It’s simply the basic encoder-decoder CNN, but without the total internal reflection mask and the rendering error map as input. And without the cost volume and latent vector optimization. The last column measures the different metrics for a model with all these optimizations: Total internal reflection mask and rendering error map are taken into account, cost volume and latent vector optimization are applied.

## Testing the model on real world data [work in progress]
Many papers on 3D reconstruction of transparent shapes use synthetic datasets. While this speeds up the acquisition of training data and reduces cost, it often leads to models that perform well on the synthetic dataset, but can’t generalize as easily to real world data. Before we compare both … I will first explain how to test the model on real world data.Taking the images of the transparent objects is pretty straight-forward, you can simply use commodity hardware like the camera in your smartphone. The other inputs are not as easy to aquire. When the authors tested their model, they created the segmentation masks manually. You might want to check out the paper “Segmenting Transparent Objects in the Wild” [7] or its successor that uses transformers [9]. The code is available here [8] and [9a]. Back to the inputs.
mirror sphere for environment map
COLMAP for viewpoint
Real world data:
How is the environment map captured
How to determine the viewing angle just with images?
## Ablation studies
The authors research the effect of using either only 5 images or a total of 20 images. Compared to a chamfer distance of 2.0x10^-4 for 10 images, we can see that reconstruction can be further improved to 1.2x10^-4 when taking 20 images for the reconstruction. Only using 5 images however increases the loss more than threefold to 6.30x10^-4.

## Comparison to SOTA [work in progress]
Quick reminder: this is deep learning research. It comes at no big surprise, that there is a newer paper [10], with a different approach, that works better. This newer paper is the successor of [4]. Image [insert image number] shows the qualitative results of this new paper. The left side presents the results of the newer paper (1st column) compared with their ground truth (2nd column). These results are displayed next to the results from "Through the looking glass". 
[] On the right side, comparing ground truth (3rd column) and shape reconstruction (4th column).
![](https://i.imgur.com/FA2KbQJ.png)
![](https://i.imgur.com/K4vB6kA.png)

It is clearly visible that the results of the paper I presented are oversmoothed. There is no space between the hand and the head of the monkey. And neither the eyes of the dog nor the monkey are visible in the reconstructions. This newer approach on the other side successfully reconstructs the eyes of both animals and clearly separates the hand from the head of the monkey. One possible reason for this oversmoothing is the average pooling in the modified PointNet++. While comparing both results, keep in mind that the underlying ground truth in both papers is slightly different. Furthermore the paper I presented, optimized for acquisition ease. The newer paper improved their acquisition ease compare to [4] but is still more restricted than the paper I presented



# Conclusion

This paper proposed a novel approach that combined different paths and provides good results

For the sake of simplicity, I left out some relevant details. If you have any questions, take a look at the paper [1], the code of the paper [2] or send me an email.
Thanks to Pengyuan and Yu-Ying Yeh for their kind support.



# References

[1] Li, Zhengqin, Yu-Ying Yeh, and Manmohan Chandraker. "Through the looking glass: neural 3D reconstruction of transparent shapes." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
[2] https://github.com/lzqsd/TransparentShapeReconstruction
[2a] https://github.com/lzqsd/TransparentShapeDataset
[3] Qian, Yiming, Minglun Gong, and Yee-Hong Yang. "Stereo-based 3D reconstruction of dynamic fluid surfaces by global optimization." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
[4] Wu, Bojian, et al. "Full 3D reconstruction of transparent objects." arXiv preprint arXiv:1805.03482 (2018).
[5] Stets, Jonathan, et al. "Single-shot analysis of refractive shape using convolutional neural networks." 2019 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE, 2019.
[6] Sajjan, Shreeyak, et al. "Clear grasp: 3d shape estimation of transparent objects for manipulation." 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020.
[7] Xie, Enze, et al. "Segmenting transparent objects in the wild." Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIII 16. Springer International Publishing, 2020.
[8] https://github.com/xieenze/Segment_Transparent_Objects
[9] Xie, Enze, et al. "Segmenting transparent object in the wild with transformer." arXiv preprint arXiv:2101.08461 (2021).
[9a] https://github.com/xieenze/Trans2Seg
[10] Lyu, Jiahui, et al. "Differentiable refraction-tracing for mesh reconstruction of transparent objects." ACM Transactions on Graphics (TOG) 39.6 (2020): 1-13.
[11] Kutulakos, Kiriakos N., and Steven M. Seitz. "A theory of shape by space carving." International journal of computer vision 38.3 (2000): 199-218.
[12] https://www.youtube.com/watch?v=cGs90KF4oTc&t=73s
[13] https://secure.math.ubc.ca/~cass/courses/m309-01a/chu/Fundamentals/reflection.htm
[14] https://secure.math.ubc.ca/~cass/courses/m309-01a/chu/Fundamentals/snell.htm
[15] https://en.wikipedia.org/wiki/Fresnel_equations
[16] https://www.youtube.com/watch?v=7LU0KcnSTc4
[17] Murase, Hiroshi. "Surface shape reconstruction of an undulating transparent object." Proceedings Third International Conference on Computer Vision, 1990



