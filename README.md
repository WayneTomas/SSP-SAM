# SSP-SAM: SAM with Semantic-Spatial Prompt for Referring Image Segmentation

<!-- <p align="center" width="100%">
<img src="ISEKAI_overview.png"  width="80%" height="80%">
</p> -->

<div>
<div align="center">
    <a href='https://scholar.google.com/citations?user=D-27eLIAAAAJ&hl=zh-CN' target='_blank'>Wei Tang<sup>*,1</sup></a>&emsp;
    <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=SVQYcYcAAAAJ' target='_blank'>Xuejing Liu<sup>&#x2709,2</sup></a>&emsp;
    <a href='https://scholar.google.com.hk/citations?user=a3FI8c4AAAAJ&hl=zh-CN' target='_blank'>Yanpeng Sun<sup>1</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=1c9oQNMAAAAJ&hl=zh-CN' target='_blank'>Rui Zhao<sup>2</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=IhYATC0AAAAJ&view_op=list_works&sortby=pubdate' target='_blank'>Fei Tan<sup>2</sup></a>&emsp;
    <a href='https://imag-njust.net/zechaoli/' target='_blank'>Zechao Li<sup>&#x2709,1</sup></a>
</div>
<div>
<div align="center">
    <sup>1</sup>Nanjing University of Science and Technology;
    <sup>2</sup>SenseTime Research&emsp;
    </br>
    <sup>&#x2709</sup> Corresponding Author
    
</div>
 
 -----------------

![](https://black.readthedocs.io/en/stable/_static/license.svg)

## Updates
- **25 June, 2023**: :boom::boom:  Our paper "SSP-SAM: SAM with Semantic-Spatial Prompt for Referring Image Segmentation" has been submitted to IEEE Transactions on Image Processing (TIP).

This repository contains the **official implementation** and **checkpoints** of the following paper:

> **Context Disentangling and Prototype Inheriting for Robust Visual Grounding**<br>
> 
>
> **Abstract:** *This paper presents SSP-SAM, an end-to-end framework that enables the Segment Anything Model (SAM) with grounding ability to tackle Referring Image Segmentation (RIS). Beyond the vanilla SAM that struggles with languages, SSP-SAM seamlessly manages such situations via a Semantic-Spatial Prompt (SSP) encoder, where images and languages are transformed into semantically enriched and spatially detailed prompts. To integrate spatial information into the semantic of the referent, we incorporate both visual and linguistic attention adapters into the SSP encoder. This process highlights the salient objects within the visual features and the discriminative phrases within the linguistic features. Such a design provides enhanced referent features for the prompt generator, leading to high-quality SSPs. Extensive experiments on widely used RIS benchmarks confirm the superiority of our method, which fully leverages SAM's segmentation capabilities. Moreover, we explore the open-vocabulary capability of SSP-SAM on PhraseCut dataset, which demonstrates improved performance in open-vocabulary scene compared to existing state-of-the-art RIS methods. The code and models will be available at https://github.com/WayneTomas/SSP-SAM once the manuscript is accepted.*

