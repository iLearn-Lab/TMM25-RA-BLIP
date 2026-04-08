<div align="center">
<h2 align="center">
    <b>RA-BLIP: Multimodal Adaptive Retrieval-Augmented Bootstrapping Language-Image Pre-training </b>
</h2>
<div>
<a target="_blank" href="#">Author&#160;Name 1</a><sup>1</sup>,
<a target="_blank" href="#">Author&#160;Name 2</a><sup>1</sup>,
<a target="_blank" href="#">Author&#160;Name 3</a><sup>2&#9993</sup>
</div>
<sup>1</sup>Institution Name 1&#160;&#160;&#160;</span>
<sup>2</sup>Institution Name 2</span>
<br />
<sup>&#9993&#160;</sup>Corresponding author&#160;&#160;</span>
<br/>
<div align="center">
    <a href="https://ieeexplore.ieee.org/iel8/6046/10844992/11125516.pdf" target="_blank">
    <img src="https://img.shields.io/badge/Paper-TMM%202025-blue" alt="IEEE TMM"></a>
    <a href="#" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
    <a href="#" target="_blank">
    <img src="https://img.shields.io/badge/Project-RA--BLIP-9cf" alt="Project Page"></a>
    <a href="#" target="_blank">
    <img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow" alt="Hugging Face Model"></a>
</div>
</div>
        
## :new: Updates


## :rocket: Abstract 
Multimodal Large Language Models (MLLMs) have recently received substantial interest, which shows their emerging potential as general-purpose models for various vision-language tasks. MLLMs involve significant external knowledge within their parameters; however, it is challenging to continually update these models with the latest knowledge, which involves huge computational costs and poor interpretability. Retrieval augmentation techniques have proven to be effective plugins for both LLMs and MLLMs. 

In this study, we propose **multimodal adaptive Retrieval-Augmented Bootstrapping Language-Image Pre-training (RA-BLIP)**, a novel retrieval-augmented framework for various MLLMs. We first leverage the question to instruct the extraction of visual information through interactions with one set of learnable queries, minimizing irrelevant interference and redundancy during retrieval and generation. Besides, we introduce a pre-trained multimodal adaptive fusion module to achieve question text-to-multimodal retrieval and integration of multimodal knowledge by projecting visual and language modalities into a unified semantic space. Furthermore, we present an **Adaptive Selection Knowledge Generation (ASKG)** strategy to train the generator to autonomously discern the relevance of retrieved knowledge, which realizes excellent denoising performance. Extensive experiments on open multimodal question-answering datasets demonstrate that RA-BLIP achieves significant performance and surpasses the state-of-the-art retrieval-augmented models.

## 🧩 Framework

<p align="center">
  <img src="./assets/framework.png" width="90%">
</p>

**Overview of RA-BLIP:** 1. **Query-Instructed Visual Extraction:** We leverage the input question to guide the extraction of relevant visual features using learnable queries. This drastically reduces visual redundancy and irrelevant interference.
2. **Multimodal Adaptive Fusion:** A pre-trained fusion module projects both visual and textual modalities into a unified semantic space, enabling seamless text-to-multimodal retrieval.
3. **Adaptive Selection Knowledge Generation (ASKG):** The generator is trained to autonomously evaluate and filter retrieved knowledge, providing highly effective denoising before the final answer generation.

## ⚙️ Getting Started

### Installation
We recommend setting up the environment using `conda`:

```shell
# Create conda environment
conda create -n rablip python=3.9
conda activate rablip

# Install PyTorch
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install required packages
git clone [https://github.com/YourOrg/RA-BLIP.git](https://github.com/YourOrg/RA-BLIP.git)
cd RA-BLIP
pip install -r requirements.txt
