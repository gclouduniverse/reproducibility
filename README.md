## For GPU recipes: https://github.com/AI-Hypercomputer/gpu-recipes
## For TPU recipes: https://github.com/AI-Hypercomputer/tpu-recipes

<br />
<br /> 
<br /> 
<br /> 
<br /> 
<br /> 

# reproducibility - DEPRECATED 
  
## Workload Reproduction Demo
This repository provides the necessary files and instructions to reproduce a specific workload to individuals outside of Google.

## Contents
- Dockerfile: Defines the Docker image required to run the workload. It specifies the base image, dependencies, and necessary configurations.
- .yaml files: Contains the configuration settings for the workload, including parameters, data sources, environment variables, and container settings.
- Instructions: steps_to_reproduce.md provides step-by-step guidance on how to setup the environment, change settings, and run the workload.

## Prerequisites
- Google Cloud GKE Cluster: Requires a configured cluster with at least 2 A3+ nodes managed with GKE.
- Docker: To build an image from the selected Dockerfile
- Helm: In steps_to_reproduce.md, helm is used to deploy a chart that includes a job definition for the workload. 
- Git: To clone this repository

## Notes
You don't have to use the instructions for reproduciblity, but you can use the values in the .yaml files and Dockerfile in your own setup.
