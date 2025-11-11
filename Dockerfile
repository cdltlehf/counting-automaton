# Image
FROM python:3.14-slim
WORKDIR /usr/local/app

# Setup an app user so the container doesn't run as the root user
RUN adduser --quiet --disabled-password --gecos "" csa-user

# Copy project and install dependencies
COPY . .
RUN pip install --upgrade pip wheel setuptools
# Install project package(s)
RUN pip install .

CMD ["python3", "-m", "cai4py.match"]

USER csa-user