FROM python:3.11

ENV APP_HOME /app
WORKDIR $APP_HOME

# Grab native files
COPY requirements.txt /app

# Install dependency packages
ENV VIRTUAL_ENV=/usr/local
RUN pip install uv && uv pip install --no-cache -r requirements.txt

# Copy source code
COPY . /app

ENV HOST=0.0.0.0
ENV LISTEN_PORT 8000
EXPOSE 8000

ENTRYPOINT ["python", "-m", "chainlit", "run", "app.py"]