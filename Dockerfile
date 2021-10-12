# Dockerfile for application server

# Install Python (not alpine due to Kaleido issue)
# https://github.com/plotly/Kaleido/issues/34
# FROM python:3.8-alpine
FROM python:3.8

# Install pip requirements
COPY server/requirements.txt /
RUN pip install -r /requirements.txt

# Copy across the application files
# (includes static folder)
COPY server/server /app

# Install airrmap as a library
COPY server/setup.py /
COPY server/airrmap /airrmap
RUN pip install -e /.

# Misc
WORKDIR /app
EXPOSE 5000:5000

# Start server
CMD [ "hypercorn", "app:app", "--bind", "0.0.0.0:5000" ]