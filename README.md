# SSP-SAM: SAM with Semantic-Spatial Prompt for Referring Expression Segmentation

<!-- <p align="center" width="100%">
<img src="ISEKAI_overview.png"  width="80%" height="80%">
</p> -->

<div>
<div align="center">
    <a href='https://scholar.google.com/citations?user=D-27eLIAAAAJ&hl=zh-CN' target='_blank'>Wei Tang<sup>*,1</sup></a>&emsp;
    <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=SVQYcYcAAAAJ' target='_blank'>Xuejing Liu<sup>&#x2709,2</sup></a>&emsp;
    <a href='https://scholar.google.com.hk/citations?user=a3FI8c4AAAAJ&hl=zh-CN' target='_blank'>Yanpeng Sun<sup>2</sup></a>&emsp;
    <a href='https://imag-njust.net/zechaoli/' target='_blank'>Zechao Li<sup>&#x2709,1</sup></a>
</div>
<div>
<div align="center">
    <sup>1</sup>Nanjing University of Science and Technology;
    <sup>2</sup>NExT++ Lab, School of Computing, National University of Singapore;
    <sup>3</sup>Institute of Computing Technology, Chinese Academy of Sciences;
    </br>
    <sup>&#x2709</sup> Corresponding Author
    
</div>
 
 -----------------

![](https://black.readthedocs.io/en/stable/_static/license.svg)

## Updates
- **20 June, 2025**: :boom::boom:  Our paper "SSP-SAM: SAM with Semantic-Spatial Prompt for Referring Referring Segmentation" has been submitted to IEEE Transactions on Circuits and Systems for Video Technology (TCSVT).

This repository contains the **official implementation** and **checkpoints** of the following paper:

> **SSP-SAM: SAM with Semantic-Spatial Prompt for Referring Image Segmentation**<br>
> 
>
> **Abstract:** *The Segment Anything Model (SAM) excels at general image segmentation but has limited ability to understand natural language, which restricts its direct application in Referring Expression Segmentation (RES). Toward this end, we propose SSP-SAM, a framework that fully utilizes SAM’s segmentation capabilities by integrating a Semantic-Spatial Prompt (SSP) encoder. Specifically, we incorporate both visual and linguistic attention adapters into the SSP encoder, which highlight salient objects within the visual features and discriminative phrases within the linguistic features. This design enhances the referent representation for the prompt generator, resulting in high-quality SSPs that enable SAM to generate precise masks guided by language. Although not specifically designed for Generalized RES (GRES), where the referent may correspond to zero, one, or multiple objects, SSP-SAM naturally supports this more flexible setting without additional modifications. Extensive experiments on widely used RES and GRES benchmarks confirm the superiority of our method. Notably, our approach generates segmentation masks of high quality, achieving strong precision even at strict thresholds such as Pr@0.9. Further evaluation on the PhraseCut dataset demonstrates improved performance in open-vocabulary scenarios compared to existing state-of-the-art RES methods. The code and models will be available at https://github.com/WayneTomas/SSP-SAM once the manuscript is accepted.*

