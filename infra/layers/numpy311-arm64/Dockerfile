FROM public.ecr.aws/lambda/python:3.11-arm64

# Install zip utility
RUN yum install -y zip && yum clean all

# Work in /opt (standard for Lambda layers)
WORKDIR /opt

# Create the layer directory structure
RUN mkdir -p python/lib/python3.11/site-packages

# Install numpy into the layer path
RUN pip install --no-cache-dir numpy==1.26.4 -t python/lib/python3.11/site-packages


# Package the layer into /layer.zip
RUN zip -r9 /layer.zip python
# The resulting layer.zip can be found at /layer.zip    
