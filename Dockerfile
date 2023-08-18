FROM continuumio/anaconda3

COPY . /
RUN conda create -n brainstormer-env python=3.8.16

# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 conda_env.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 conda_env.yml | cut -d' ' -f2)/bin:$PATH

RUN pip install -r environment/requirements.txt

ENV FLASK_APP=app.py
ENV FLASK_DEBUG=true
EXPOSE 8084

CMD ["flask", "run", "--host=0.0.0.0", "--port=8084"]

