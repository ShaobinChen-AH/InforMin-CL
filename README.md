# An Information Minimization Based Contrastive Learning Model for Unsupervised Sentence Embeddings Learning.

This repository contains the code and pre-trained models for our paper [An Information Minimization Based Contrastive Learning Model for
Unsupervised Sentence Embeddings Learning](https://arxiv.org/abs/2209.10951).<br>
\*\*\*\*\*\**\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\***Updates**\*************************************<br>
<li>9/25: Recently I am busy applying for a PhD program. Some information will be updated later.
<li>9/16: We release our code.
<li>8/27: Our paper has been accepted to COLING 2022!</li>

<strong>Quick Links</strong><br>
  ---
<li>Overview</li>
<li>Train InforMin-CL</li>
<ul>
<li>Requirements</li>
</ul>
<li>Bugs and questions</li>
<li>Citation</li><br>

<strong>Overview</strong>
  ---
We propose a contrastive learning model, InforMin-CL that discards the redundant information during the pre-training phase. InforMin-CL keeps important information and forgets redundant information by contrast and reconstruction operations. The following figure is an illustration of our model.<br>
![model png](https://user-images.githubusercontent.com/51829876/190571095-ef35e783-dd96-4e41-b4fe-185f735225e1.jpg)

<strong>Train InforMin-CL</strong>
  ---
In the following section, we describe how to train a InforMin-CL model by using our code.<br>


<strong>Requirements</strong><br>

First, install PyTorch by following the instructions from the [the official website](https://pytorch.org/). To faithfully reproduce our resutls, please use the correct <code>1.7.1</code> version corresponding to your platforms/CUDA versions. PyTorch version higher than <code>1.7.1</code> should also work. For example, if you use Linux and <strong>CUDA11</strong>, install PyTorch by the following command,<br>

<code>conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch</code><br>

if you instead use <strong>CUDA</strong><code><11</code> or <strong>CPU</strong>, install PyTorch by the following command,<br>

<code>pip install torch==1.7.1</code><br>

Then run the following script to install the remaining dependencies,<br>

<code>pip install -r requirements.txt</code><br>

<strong>Evaluation</strong><br>

Our evaluation code for sentence embeddings is based on a modified version of [SentEval](https://github.com/facebookresearch/SentEval). It evaluates sentence embeddings on unsupervised (semantic textual similarity (STS)) tasks and supervised tasks. For unsupervised tasks, our evaluation takes the "all" setting, and report Spearman's correlation.<br>

Before evaluation, please download the evaluation datasets by running<br>
<code>
  cd SentEval/data/downstream/  
  bash download_dataset.h
</code>

Then come back to the root directory, you can evaluate any <code>transformers</code> -based pre-trained models using our evaluation code. For example,<br>
<code>
  python evaluation.py \
    --model_name_or_path informin-cl-bert-base-uncased \
    --pooler cls \
    --text_set sts \
    --mode test \
</code>

<strong>Training</strong><br>
<br>
<code>
python train.py \
  --model_name_or_path bert-base-uncased \
</code>


<strong>Bugs or questions?</strong><br>
  ---
If you have any questions related to the code or the paper, feel free to contact with Shaobin (<code>chenshaobin000001@gmail.com</code>). If you enconuter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can give you a hand!<br>
  
<strong>Citation</strong><br>
  ---
Please cite our paper if you use InforMin-CL in your work:<br>
<code>
    @inproceedings{chen2022informin-cl,\
        title={An Information Minimization Contrastive Learning Model for Unsupervised Sentence Embeddings Learning},\
        author={Chen, Shaobin and Zhou, Jie and Sun, Yuling and He Liang},\
      booktitle={International Conference of Computational Linguistics (COLING)},\
      year={2022}}
</code>
