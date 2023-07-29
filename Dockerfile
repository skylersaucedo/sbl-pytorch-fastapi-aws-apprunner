FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11
WORKDIR /code
COPY . /code
RUN pip install --upgrade pip &&\
   pip install --no-cache-dir --upgrade -r /code/requirements.txt
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

# FROM ubuntu:18.04
# ENV LC_ALL=C
# RUN apt-get update -y && apt-get install -y python3-pip python3-dev build-essential pkg-config

# # Install extras
# #COPY requirements.yml /requirements.yml
# COPY requirements.txt /requirements.txt
# COPY . /code/
# # If you are using a py27 image, change this to py27
# #RUN /bin/bash -c ". activate py36 && conda env update -f=/requirements.yml"
# CMD ["bash"]
# RUN pip3 install -r /requirements.txt
# RUN python app.py

# #CMD [ "flask", "run", "--host=0.0.0.0" ]
