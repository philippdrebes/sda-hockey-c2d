FROM python:3.11.5-bookworm

# Install dependencies
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# Copy source code
COPY . .
