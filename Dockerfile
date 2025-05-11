FROM eclipse-temurin:8-jdk

RUN mkdir -p /usr/src/app
COPY . /usr/src/app/
WORKDIR /usr/src/app

# Download gradle wrapper and build cache
RUN chmod +x gradlew
RUN ./gradlew build --no-daemon

ENTRYPOINT ["./gradlew", "run", "--no-daemon"]