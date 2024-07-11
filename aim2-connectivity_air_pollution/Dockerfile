FROM ubuntu:22.04

WORKDIR /home

RUN mkdir /tmp/abcdWrangler
RUN mkdir /tmp/IDConn
RUN mkdir /home/code
RUN apt-get update && apt-get install -y python3.9 python3-pip 

COPY Data/abcdWrangler /home/abcdWrangler
COPY Projects/IDConn/IDConn /home/IDConn


WORKDIR /home/IDConn
RUN pip install . #&& pip install jupyter -U && pip install jupyterlab
WORKDIR /

#EXPOSE 8888

#ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]
CMD python3 /home/code/1.1nbs_predict-ABCD-baseline.py