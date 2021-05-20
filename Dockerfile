ARG IMAGE_TAG

# ------ Base -----
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel as base_baal

RUN pip install --upgrade pip
COPY requirements.txt /app/baal/requirements.txt
RUN pip install -r /app/baal/requirements.txt
COPY . /app/baal
WORKDIR /app/baal
RUN pip install -e .[nlp] --no-use-pep517

# ---- test -----
# we need to install test dependencies before, so we cannot use 'base_baal' as base image
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel as test_baal

WORKDIR /app/baal

COPY ./test-requirements.txt /app/baal/test-requirements.txt
COPY ./requirements.txt /app/baal/requirements.txt

RUN pip install -r /app/baal/test-requirements.txt
RUN pip install -r /app/baal/requirements.txt
COPY --from=base_baal /app/baal .
RUN pip install -e .[nlp] --no-use-pep517

# ---- release image ----
FROM base_baal as release