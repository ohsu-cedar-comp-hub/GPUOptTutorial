# GPU Optimization Tutorial


## **Introduction**

#### **Background**

This tutorial is based off of Layaa’s script that splits whole slide images (.svs) into 256 x 256 tiles. Tiles are assessed and those with poor contrast and variation (aka likely background) are filtered out. Remaining tiles’ embeddings and coordinates are ran through the pretrained GigaPath model which will output slide-level embeddings. These embeddings along with the tile’s position are captured in a tsv for future processing. 

Layaa was running this script on 4 huge datasets, with each dataset containing ~ 1000-3000 images. Images were split into batches of five and each image was very big, with a height and width easily in the range of 15k to over 100k. 

This script was ran as an SBATCH job and each job was one batch of .svs images. Jobs were submitted using a job array. 

#### **Getting Started**

This tutorial assumes you have conda and you have knowledge on how to pull from a Git repository. 

If you’re unsure, refer to: 

https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

https://docs.github.com/en/get-started/using-git/getting-changes-from-a-remote-repository

#### **Data Input** 

For the purposes of this tutorial, we have cropped images to a more manageable size of 15k x 15k. We will be using 2 batches with each batch consisting of 2 images to also demonstrate the utility of the job array. An example of what one image looks like is below: 

![image.png](assets/example.png)

## **Running the Tutorial**

#### **Setting Up**

First, you will pull all the files from the GPU Opt github: 

```
git clone https://github.com/ohsu-cedar-comp-hub/GPUOptTutorial.git
cd gpu_opt
```

Confirm that your current working directory is the gpu_opt directory. 

Now, we will set up the correct environment: 

Follow install instructions from gigapath github README. https://github.com/prov-gigapath/prov-gigapath 

Then run this block of code below, making sure to install everything in the same gigapath environment.

(You may be able to skip uninstalling and reinstalling torch modules if you didn’t run into any issues installing gigapath from github). 

```
pip uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

conda install anaconda::pandas
conda install conda-forge::timm
conda install anaconda::tifffile
pip install tiler

```

#### **Pulling the Data** 
For this tutorial, we will be using 2 batches of 2 BRCA images with each image being 15k x 15k pixels. 
Batch 1 is titled TCGA_BRCA-batch_1. Batch 2 is titled TCGA_BRCA-batch_2. 

Pull the data by creating a symbolic link. 

```
cd gpu_opt
ln -s /home/exacloud/gscratch/CEDAR/chaoe/gpu_opt/TCGA-BRCA .
```


#### **Launching the Tutorial**

Our launch script is titled launch.sh. We will use these already present sbatch parameters: 

```bash
#!/bin/bash
#SBATCH --partition gpu
#SBATCH --account CEDAR
#SBATCH --gres=gpu:a40:1    
#SBATCH --array=1-2%2
#SBATCH --cpus-per-task 1
#SBATCH --mem 20G
#SBATCH --time 1:00:00
#SBATCH --job-name gpu_opt_tut
```

Here we are requesting 1 a40 GPU, 1 CPU, 20G of memory and setting a time limit of 1 hour. We also set up a job array of 2 tasks and we want both to run in parallel. 

Inside the launch script, we will call the following: 

```bash
python scripts/script.py -id TCGA-BRCA/TCGA-BRCA-batch_${SLURM_ARRAY_TASK_ID} -hf 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 -lf log/TCGA-BRCA/TCGA-BRCA-batch_${SLURM_ARRAY_TASK_ID} -o results/ $CACHE_ARG
```

We are running the script (script.py) with its required arguments:  

The image directory (-id) is TCGA-BRCA/TCGA-BRCA-batch_${SLURM_ARRAY_TASK_ID}. We are using a job array, so there are two image directories we are running in parallel. 

The hugging face token (-hf) is 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. 

I’ve also set the path for my log files (-lf) as log/TCGA-BRCA/TCGA-BRCA-batch_${SLURM_ARRAY_TASK_ID}.log. 

I’ve also set an output directory in results/ . 

Lastly, there is a cache argument that can be filled in if it is provided during launch of the shell script. This is for the location of the HuggingFace cache. Through my testing, I like to use a cache directory in my gscratch. If you leave it blank, it will just go to your default which is at RDS! 

TIP: Want to see more information regarding why I put HF cache in gscratch? Move to [**Changing Location of Cache** ](#changing-location-of-cache)

TIP: Want a more detailed breakdown of what’s happening in script.py? Move to [**Detailed Breakdown of Script (script.py)** ](#detailed-breakdown-of-script-scriptpy). 


Now, we will launch the job array by running this command: 

`sbatch scripts/launch.sh [insert cache location here if desired]` . 


After launching, you can confirm that the job array is functioning properly by using `squeue` and checking the log file. 

## **Tracking and Optimizing the Jobs**

Often, we have no idea the amount of resources our jobs need. A good litmus test is to track the usage and efficiencies of the resources requested on a smaller test job. In this case, that is this tutorial. 

#### **GPU Usage and Optimization** 

While your job is running, you will need to ssh into the compute node(s) your job is running on and run nvidia-smi. This is a great easy command to confirm that you are using your GPU(s) and to see how much of your GPU(s) are being used. Refer to [**Why Nvidia Smi?** ](#why-nvidia-smi) for a small example of why this is important!

Use `watch nvidia-smi` to get real time updates as the job ran. 

For example, this is what I saw when I did this a few minutes into my jobs: 

![image.png](assets/image2.png)

From a quick glance, we can confirm that we are using our GPU! We can tell that we are using CUDA, and that we are using the a40 GPU. 

Looking at GPU utilization, we can see the percentage of time the streaming multiprocessors (SMs) were running over a sampled time period. In this moment, 79% were used. 

Because the GPU utilization varies over a sampled time period and will change depending on what’s occurring in the job, this number should be taken with a grain of salt. The main focus is whether the GPU(s) is actually being used at all and if the GPU utilization is very low. 

If your GPU utilization is very low, that means that your GPU is not being fully utilized or is idle. 

There could be many causes behind this, but a common one is that the GPU is being bottlenecked by slow data loading. The utilization can also be bottlenecked by inefficient code, overloaded CPU(s) and/or memory limits. 

One solution that people default to is to increase the batch size to potentially improve your GPU optimization. I would approach that solution with caution however as that also increases the amount of data you will be loading in, so CPU optimization is also really important. 

TIP: Check if you have slow data loading by using the time module to track it! 

#### **CPU Usage and Optimization** 

As mentioned above, GPU efficiency is also heavily dependent on CPU efficiency and optimizing your CPUs are a lot easier! 

You can track the usage and efficiency of your CPU(s) from a past job using a handy SLURM job assessment tool that can be obtained here: https://github.com/ohsu-cedar-comp-hub/SlurmStats. This tool also displays time and memory efficiencies. 

This tool generates a report that allows you to check your CPU, memory and time efficiency. In the case of this tutorial, I got these stats back: 

![image.png](assets/image3.png)

So from the above, the requested 1 CPU was appropriate. The 20 GB memory requested was also appropriate. The only thing that could use a big change is the requested time limit since these jobs only took around 5 minutes. 

TIP: A good rule of thumb is to aim for > 50% efficiency! 

This tool is also useful to keep track of how much time the job took (via Elapsed column) and is utilized later when I compared other factors! 

## **Other Information**

#### **Detailed Breakdown of Script (script.py)** 

INPUT: Path to Directory of Whole Slide Images (.svs), Hugging Face Token, Path to Log File, Path to Results Directory 

    For each image in the image directory: 

        New ImageCropTileFilter object is created. 

        - Image is read in using Tifffile library.
        - Time taken to read image is printed to log file.
        - Obtain information of the cancer type, image file name, sub id etc from the image file name.

        Load the gigapath model using the hugging face token. 

        - Time taken to load model is printed to log file.
        - Explicitly set torch device to cuda.
        - Create an array of transformations to be applied later
            - resize image so shorter side is 256 pixels
            - crops a 224 x 224 square from image’s center
            - converts image into PyTorch tensor
            - normalizes RGB using predetermined mean and standard deviation

        Crop the image to ensure its dimensions are divisible by 256. 

        Tile the image into 256 x 256 tiles. 

        For each tile: 

            Record the tile’s position coordinates in the image. 

            Record the number of unique pixel values in the tile and their occurrences as an array. 

            Calculate the 5th percentile and 50th percentile of pixel values. 

            Filter the tile based on if it’s likely to be background or if it has the tissue: 

            If likely tissue aka has lots of contrast, tile must pass these conditions: 

            - smallest unique pixel value < 135 (some dark pixels)
            - largest unique pixel value ≥ 255 (some bright pixels)
            - 5th percentile is < 162 and 50th percentile < 225 (not too bright/washed out)

            Tile marked as tissue is converted into PIL Image object. 

            Array of transformations is applied and tile becomes an RGB image. 

            Tile is ran through gigapath model. 

            The output (slide-level embeddings) and metadata of tile is saved to a dataframe in the results directory. 

            If likely background, tile is ignored and not saved. 

OUTPUT: Log File, Dataframe of Processed (Likely Tissue) Tiles 

#### **Why nvidia-smi?** 

When running a job that requires a GPU, it is imperative to confirm that you are actually using your GPU(s)! 

To demonstrate this, I’ve taken the original script and modified it slightly so that it won’t use the GPU as the pytorch tensor isn’t explicitly switched to cuda. This new script is called mini_script_error.py with the launch script being mini_error.sh. 

I’m only using one image, with size of 5k x 5k as input as I expect this job to take a lot longer. 

I launched using `sbatch scripts/mini_error.sh`

After submitting the job, confirming using `squeue`, I ran `watch nvidia-smi` to track GPU usage and as you can see below, no GPU was being utilized. 

![image.png](assets/image4.png)

The job was still able to complete, it just took a lot longer than it would have! 

The job took around 15 minutes, while with GPU, it would have taken only 43 seconds! 

![image.png](assets/image5.png)

Since the job was still able to finish with only CPU, it may not have raised any suspicion that it wasn’t using GPU. But, it would have been a lot faster if it had! This is why it’s important to ensure that the resources we request are actually being used! 

If this small image was expanded back to its original whole slide image size, we can quickly see just how drastic the difference between using only the CPU and using both the CPU and GPU is! The difference is even more drastic if we imagine that we have a full dataset of this image to process! 

For the full dataset, it is 622 batches of 5 images. These numbers were chosen to directly match the number of BRCA images Layaa had. 

![image.png](assets/image6.png)

This visual is under two assumptions: 

1. As the small image is expanded back, the computation time and real time it takes increases proportionally to the new image size. 
2. All images in each batch and in the entire BRCA dataset are the same. 

While these assumptions indicate that the RT (real time) and CT (computation time) are likely exaggerated, this visual still gives a good look at how drastic a 20x difference can be when you apply it to your real workflow. 

This difference was only 20x for this analysis but could very much be a lot more depending on how busy the cluster is, your analysis and the data you’re working with!

It is highly recommended to use a small test job to start off with to test that the script runs and that the resources requested are being used!

#### **Using LegacyGPU Partition** 

Because this tutorial is not computationally intensive, I wanted to test the effect of using a less powerful gpu via the legacygpu partition. 

Below, is an output I see when I run `watch nvidia-smi`  . 

![image.png](assets/image7.png)

Remember that the GPU utilization is variable and dependent on where the job is at the moment, so the utilization needs to be taken with a grain of salt. 

From looking at the SLURM job assessment tool and using the time module to log the time for loading the data, and loading the model, we can see that the legacygpu partition does take ~2x longer. 

![image.png](assets/image8.png)

This is likely because legacygpu has older GPUs and older CPUs. We can compare the CPU versions below: 

![image.png](assets/image9.png)

![image.png](assets/image10.png)

Refer to the below table to learn more about the different gpu partitions and their specs. 

![image.png](assets/image11.png)

Because it does take a job longer to run on the legacygpu partition, I’d recommend using it for jobs that aren’t urgent and if the wait is too long for the gpu partition. 

Alternatively, it could be useful to start off running your job on the legacyGPU partition first, and then, if you get a CUDA out of memory error, you could move up to the a40 GPUs in the gpu partition. 

TIP: To check how busy these partitions are, use sinfo!

#### **Changing Location of Cache**

The huggingface cache by default is present in your RDS at /home/users/yourname/.cache/huggingface/hub. 

You can change where you want the cache to be when you first load in the model so what would happen if the cache was located in gscratch? 

Testing with the gpu partition first, we use the time module to track the loading difference and we can view it in the log files. 

First from loading with the default, we can see that loading the model for the first time takes 53 seconds and then the second time it takes around 14 seconds. 

![image.png](assets/image12.png)

Loading with the cache directory in gscratch makes it a lot easier to access so now loading the model takes less than 20 seconds. 

![image.png](assets/image13.png)

We see a similar advantage with the legacygpu partition, although the advantage is less pronounced. 

Again, loading from the default takes almost a minute and a half the first time. In contrast, loading with the cache in gscratch takes a little more than a minute the first time. 

![image.png](assets/image14.png)

![image.png](assets/image15.png)

NOTE: Take the time saved here with a grain of salt. How much faster it is to load on gscratch is heavily dependent on how busy RDS is at a given time. The busier RDS is, the longer it takes to load in RDS, and so loading on gscratch can be a lot faster in comparison. 

Nonetheless, gscratch is a space dedicated for fast loading and access of data, so it’s good practice to put your HF cache here. 

#### **General Tips for Lazy Loading** 

Depending on the project and the data used, you can speed up the data loading. 

For images, if you already know the coordinates of the portions you need, you can load in just the portions of the images you need rather than the entire image. To find the coordinates, you can use software like ImageJ to load the image first. 

For example, if you have the polygon coordinates of the gland of interest, you can create a bounding box and that will be the portion of the image you load in. 

In general, it is best for images to be saved as .ome.tiff as this format saves the image as multiple blocks and is therefore easier to lazy load. 

Other good file formats are .h5 and .anndata.