# GPU Optimization Tutorial

## Please go to GPUOptTutorial.md to view the full tutorial on this!! 


**Background**

This tutorial is based off of Layaa’s script that splits whole slide images (.svs) into 256 x 256 tiles. Tiles are assessed and those with poor contrast and variation (aka likely background) are filtered out. Remaining tiles’ embeddings and coordinates are ran through the pretrained GigaPath model which will output slide-level embeddings. These embeddings along with the tile’s position are captured in a tsv for future processing. 

Layaa was running this script on 4 huge datasets, with each dataset containing ~ 1000-3000 images. Images were split into batches of five and each image was very big, with a height and width easily in the range of 15k to over 100k. 

This script was ran as an SBATCH job and each job was one batch of .svs images. Jobs were submitted using a job array.

**Basic Setup** 

Clone this repo by doing: 

```
git clone https://github.com/ohsu-cedar-comp-hub/GPUOptTutorial.git
cd gpu_opt
```



