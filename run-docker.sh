docker build -t classy-cloud .

docker run --mount type=bind,source="$(pwd)",target=/app/ classy-cloud all