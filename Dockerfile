FROM python:3.8-slim

RUN groupadd -r mmmtuser && useradd -r -g mmmtuser mmmtuser

RUN apt-get update
RUN apt-get -y install git
RUN apt-get -y install gcc

WORKDIR /home/mmmtuser

RUN chown mmmtuser:mmmtuser /home/mmmtuser

USER mmmtuser

ARG GIT_TOKEN
RUN pip3 install "git+https://$GIT_TOKEN@github.com/BiomedSciAI/multimodal-model-toolkit"

CMD ["bash"]
