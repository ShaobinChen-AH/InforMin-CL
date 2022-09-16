# An Information Minimization Based Contrastive Learning Model for Unsupervised Sentence Embeddings Learning.

***

This repository contains the code and pre-trained models for our paper.<br>
\*\*\*\*\*\**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***Updates**\*************************************<br>
<li>8/27: Our paper has been accepted to COLING 2022!</li>
<strong>Quick Links</strong><br>

***

<li>Overview</li>
<li>Overview</li>

Overview<br>

***

We propose a contrastive learning model, InforMin-CL that discards the redundant information during the pre-training phase. InforMin-CL keeps important information and forgets redundant information by contrast and reconstruction operations. The following figure is an illustration of our model.<br>
![model png](https://user-images.githubusercontent.com/51829876/190571095-ef35e783-dd96-4e41-b4fe-185f735225e1.jpg)
<strong>Train InforMin-CL</strong>
***
Tn the following section, we describe how to train a InforMin-CL model by using our code.<br>
<strong>Requirements</strong><br>
First, install PyTorch by following the instructions from the [the official website](https://pytorch.org/). To faithfully reproduce our resutls, please use the correct <code>1.7.1</code> version corresponding to your platforms/CUDA versions. PyTorch version higher than <code>1.7.1</code> should also work. For example, if you use Linux and <strong>CUDA11</strong>, install PyTorch by the following command,<br>
<code>conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch</code><br>
if you instead use <strong>CUDA</strong><code><11</code> or <strong>CPU</strong>, install PyTorch by the following command,<br>
<code>pip install torch==1.7.1</code><br>
Then run the following script to install the remaining dependencies,<br>
<code>pip install -r requirements.txt</code><br>

<strong>Bugs or questions?</strong>
