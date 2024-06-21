# Sim_Funcky_Pipeline

Sim_Funcky_Pipeline is a pipeline designed to preprocess BOLD Functional MRIs for visualization and further analysis. This tool also calculates the average signal intensity per ROI in a given volumetric atlas and returns a pairwise similarity matrix. 

## Overview

The preprocessing pipeline closely follows the recommendations outlined by Power et al 2014:

> Power JD, Mitra A, Laumann TO, Snyder AZ, Schlaggar BL, Petersen SE. Methods to detect, characterize, and remove motion artifact in resting state fMRI. Neuroimage. 2014 Jan 1;84:320-41. doi: 10.1016/j.neuroimage.2013.08.048. Epub 2013 Aug 29. PMID: 23994314; PMCID: PMC3849338.


These steps include (but are not limited to):

### Preprocessing Steps
- **Reorientation**: Aligning the MRI images to a standard orientation.
- **Skull Stripping**: Removing non-brain tissues from the MRI images.
- **Rigid Body Realignment**: Correcting for patient movement during the scan.
- **Median 1000 Normalization**: Normalizing the voxel intensity distribution to its median.
- **Demeaning**: Centering the data around zero.
- **Motion Parameter Regression**: Removing 6 motion parameters, their squares, derivatives, and derivative's squares (total of 24).
- **Bandpass Filtering**: Applying a bandpass filter to the data.
- **Spatial Smoothin**g: Applying a smoothing filter to the data.
- **Registration**: Aligning an atlas to the patient space for standardized anatomical referencing.
- **Scrubbing: Censoring** frames with excessive motion.
- **Average Regional Signal and Similarity Calculation**: Calculating and collecting volume and first-order radiomics features per ROI.

## Features

- Written primarily in Python.
- Utilizes the Nipype library for workflow management.
- Employs tools such as FSL and ANTs for processing tasks.
- Compatible with the Brain Imaging Data Structure (BIDS) Standard.

## Inputs and Outputs

### Inputs
At minimum:
- **Parent Data Directory**: The directory containing the MRI data.
- **Subject ID**: Identifier for the subject/patient.
- **Output Directory**: Directory where the processed outputs will be saved.

### Outputs

1. **Preprocessed Patient BOLD**: The MRI after all preprocessing steps.
2. **Atlas in Patient Space**: An atlas registered to the patient’s anatomical space.
3. **Average regional signal intensity and similarity matrix**

## Usage Instructions

### Running the Tool Directly

To use this tool directly, run the following command:
```
python3 Pipeline.py -p [data_dir_path] -sid [subject-id] -o [output_path] -tem [template_path] -seg [segment_path]
```

### Using Docker (Recommended)

Using Docker is recommended to simplify the installation of necessary dependencies (including FSL, ANTs, and relevant Python libraries). There are two ways to use Docker: building and running the container locally, or using a prebuilt Docker image from Docker Hub.

**Option 1: Building and Running the Container Locally**
1. Clone the Git repository and navigate to the directory containing the Dockerfile.
2. Build the Docker container:
```
docker build -t my_Sim_Funky_Pipeline_container .
```
3. Run the Docker container:
```
docker run -v [data_dir_path]:/data/my_data -v [output_path]:/data/output --rm -u $UID:$UID my_Sim_Funky_Pipeline_container -p /data/my_data -sid [subject-id] -o [output_path] 
```

Explanation of Docker Flags:

```--rm```: Automatically removes the container after it exits. <br>
```-v```: Mounts your data and output directories into the Docker container.<br>
```-u``` $UID:$UID: Runs the Docker container as the same user as on the host machine to avoid file permission issues.

*Notes:
If the output directory is a subdirectory of the data directory (e.g., \[data_dir_path\]/derivatives), you only need to mount the data directory once and provide the output path relative to the mounted data directory (e.g., /data/my_data/derivatives). Output files will be accessible on the host machine from where the output path was mounted. Default anatomical templates and atlas (MNI152 and AALv3_CombinedThalami respectively) are included so the -tem and -seg flags are optional.*

**Option 2: Using a Prebuilt Docker Image**
1. Pull the prebuilt Docker image from Docker Hub:
```
docker pull jor115/tbd
```
2. Run the Docker container using the pulled image:
```
docker run -v [data_dir_path]:/data/my_data -v [output_path]:/data/output --rm -u $UID:$UID jor115/tbd -p /data/my_data -sid [subject-id] -o [output_path] 
```
*Note: The docker run command is identical to the one used for running a locally built container, but you do not need to download the source code or build the container locally.*

## Development Status

This tool is a work in progress. It is currently fully functional but is undergoing refinement to enhance user flexibility and input handling.

## Feedback and Contributions

If you have any concerns or suggestions regarding the code or its implementation, please contact the authors at pirc@chp.edu.

## Acknowledgement

The primary authors of this work are Joy Roy and Rafael Ceschin, both members of the Department of Biomedical Informatics, University of Pittsburgh School of Medicine and the Pediatric Imaging Research Center, Department of Radiology, Children's Hospital of Pittsburgh.
