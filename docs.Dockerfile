FROM python:3.7

RUN apt-get update && apt-get install -y -q pandoc

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
ADD documentation-requirements.txt documentation-requirements.txt
RUN pip install -r documentation-requirements.txt

ADD . .
RUN pip install .
RUN make --directory docs/ html

FROM nginx:stable

COPY --from=0 docs/_build/html /var/www/
