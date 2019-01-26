FROM chainer/chainer:v5.1.0-python3

RUN pip3 install --upgrade pip
RUN pip3 install boto3

WORKDIR /src
