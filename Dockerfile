# Dockerfile for application server

# ---- UI, build static files (Stage 1) ----

# Python is required for some npm installs
FROM python:3.8 as stage1

# Install node and parcel
# Adapted from: https://stackoverflow.com/questions/36399848/install-node-in-dockerfile/67491580#67491580
RUN apt-get update && apt-get install -y \
    software-properties-common \
    npm
RUN npm install npm@latest -g && \
    npm install n -g && \
    n latest  
#RUN npm install -g parcel@2.0.0

# Copy across ui source files
COPY ui /ui

# Install npm packages
WORKDIR /ui
RUN npm install

# Parcel compile to server/static folder
WORKDIR /
RUN mkdir /tmp/build_ui

#RUN npx parcel build ui/src/index.html --no-scope-hoist --dist-dir /tmp/build_ui 
COPY server/server/static /tmp/build_ui

# ---- Server (Stage 2) ----

## Install Python (not alpine due to Kaleido issue)
## https://github.com/plotly/Kaleido/issues/34
## -slim also fails with pip install editdistance / polyleven? (gcc issues)
## FROM python:3.8-alpine
FROM python:3.8

# Install pip requirements
COPY server/requirements.txt /
RUN pip install -r /requirements.txt

# Copy across the application files
RUN mkdir -p /app
COPY server/server/*.* /app

# Install airrmap as a library
WORKDIR /
COPY server/setup.py /
COPY server/airrmap /airrmap
RUN pip install -e /.

# Copy across static files from stage 1
COPY --from=stage1 /tmp/build_ui /app/static


# ---- Application Server ----
WORKDIR /app
EXPOSE 5000:5000

# Start server
CMD [ "hypercorn", "app:app", "--bind", "0.0.0.0:5000" ]