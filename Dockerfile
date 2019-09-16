FROM python:3.7
RUN adduser --quiet --disabled-password qtuser


ADD requirements.txt /setup.py  /app/
COPY test /app/test
COPY RBFN /app/RBFN
COPY examples /app/examples

WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install pytest
RUN pip install pytest-cov
# Install Python-Interface
RUN python setup.py install
RUN pytest --cov=RBFN test/test.py
RUN python RBFN --help
ADD . /app



