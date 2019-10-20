FROM tensorflow/tensorflow:1.14.0-py3

ADD . /root
WORKDIR /root

RUN apt-get autoclean
RUN apt-get update
RUN apt-get -y install wget
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
