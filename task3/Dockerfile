FROM --platform=linux/amd64 pytorch/pytorch AS example-algorithm-amd64
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

# Install system dependencies for PyVips (as root)
USER root
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       libvips-dev \
       pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
# COPY --chown=user:user resources /opt/app/resources

# You can add any Python dependencies to requirements.txt
RUN python -m pip install --no-cache-dir --prefer-binary pyvips
RUN pip install --upgrade pip setuptools wheel --progress-bar off && \
    pip install pandas --progress-bar off --no-cache-dir
RUN python -m pip install --upgrade pip setuptools wheel --progress-bar off && \
    pip install --no-cache-dir --disable-pip-version-check --progress-bar off -r /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user clinical_data_process.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
