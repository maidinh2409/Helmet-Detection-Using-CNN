# Helmet-Wearing-Object-Classification-Using-Deep-Learning

*[Try the model on Hugging Face Spaces](https://huggingface.co/spaces/demile2409/CSIS-3290-Helmet-Image-Classification)*

## PROJECT OVERVIEW

**<ins>Description:</ins>** This project was inspired by my Vietnamese heritage, where motorcycles are deeply embedded in everyday life. Growing up in Vietnam, I witnessed both the ubiquity of motorbikes and the critical importance of helmet usage—yet also the risks when safety compliance is overlooked. That cultural context sparked my interest in creating a model that could automatically detect whether someone is wearing a helmet in an image.

This project was developed as part of a course at Douglas College and laid the groundwork for future computer vision projects with real-world relevance and social impact.

I built and trained a Convolutional Neural Network (CNN) on a custom dataset, enabling the model to distinguish between helmet and non-helmet cases with high accuracy. The final product is deployed as an interactive web app for real-time testing.

**<ins>Purpose:</ins>** To apply deep learning to a practical safety compliance problem and build a deployable solution for environments like construction sites and traffic monitoring.

**<ins>Expected insights:</ins>** Demonstrates how CNNs can effectively detect safety gear in real-world imagery and highlights the process of building an end-to-end ML product—from training to deployment.

**<ins>Tech Stack:</ins>**

- Languages: Python
- Models: Deep Neural Network (DNN), Convolutional Neural Network (CNN)
- Libraries & Tools: PyTorch, OpenCV, Gradio, Google Colab

## MODEL EVALUATION

![image](https://github.com/user-attachments/assets/73ae6135-b79f-4c7a-afa0-c67094961bde)

**Key Metric:**

Given that the dataset exhibits class imbalance, relying solely on accuracy may not provide a true reflection of model performance across both classes. If the goal of this image classification task is to serve as a foundation for helmet detection applications — such as in factories or traffic monitoring systems — recall becomes the most appropriate evaluation metric. Recall focuses on the model's ability to correctly identify all positive cases (e.g., individuals without helmets), which is critical for ensuring safety in high-risk environments.

**Key observation:**

From the results, it is evident that CNN substantially outperforms MLP. Several technical factors contribute to this:

* MLP flattens the image into a 1D vector, discarding the spatial structure (e.g., how edges, corners, and textures are arranged). CNN maintains the 2D spatial layout, allowing the model to detect local patterns (such as the shape of a helmet) irrespective of position or orientation
* A fully connected MLP on 256×256×3 images would need millions of parameters, making it highly prone to overfitting with small datasets. CNNs use shared kernels (filters) across the image, drastically reducing the number of parameters and making learning more robust.

![image](https://github.com/user-attachments/assets/d656e57f-a3a6-4931-bfa7-0bbfb118c513)

**Important Challenges Identified**

While CNN models achieve significantly better performance, out of 71 images labeled as No Helmet, 14 were incorrectly predicted as Helmet, representing approximately 19.7% of No Helmet cases. This is considered the worst-case error for this task, as failing to detect individuals without helmets could compromise safety. It is important to acknowledge issues of detecting No Helmet as Helmet label, which might be because of:

* **Feature Similarity between Helmet and No Helmet:** Upon testing, it was observed that the model correctly classifies individuals wearing distinct objects like caps or those with bright-colored long hair (e.g., blonde) as No Helmet. However, it struggles with individuals having dark short hair or bun hairstyles, often misclassifying them as wearing helmets. This suggests that the model excels when the object structure visibly differs from a helmet but struggles when the structure is visually similar.
* **Data quality:** The dataset was cropped from an object detection dataset. As a result, many cropped images suffer from issues such as blurriness, poor lighting, or a lack of clear structural distinction between helmet and no helmet instances. This likely contributes to the model’s difficulty in differentiating subtle visual cues.

*Due to the prediction challenges discussed above, I have also provided a separate folder containing external test images alongside this submission. If you wish to test using your own images, please screenshot and crop the object to include only the head region, as the entire dataset used for training was based on cropped head images.*

*[Try the model on Hugging Face Spaces](https://huggingface.co/spaces/demile2409/CSIS-3290-Helmet-Image-Classification)*
