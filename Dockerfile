# Dockerfile for Brain Tumor Classification - CoLlAGe Extraction
# Based on radxtools/collageradiomics-pip with additional dependencies

FROM radxtools/collageradiomics-pip:latest

# Install required Python packages at correct versions
# numpy 1.19.0 required by CoLlAGe/mahotas
# pandas 1.3.5 is last version compatible with numpy 1.19.x
RUN pip install --no-cache-dir \
    numpy==1.19.0 \
    pyyaml==6.0.3 \
    nibabel==3.2.1 \
    pandas==1.3.5

# Set working directory
WORKDIR /data

# Default command (can be overridden)
CMD ["/bin/bash"]
