FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/plasmax/.venv \
    PATH="/opt/plasmax/.venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pin the package manager independently of the base image.
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/
RUN ln -s /usr/bin/python3 /usr/local/bin/python

WORKDIR /workspace

# Docker is repository-only training infrastructure, so install the locked
# environment package plus the clone-only research dependency group.
COPY . .
RUN uv sync --frozen --no-dev --group research \
    && uv pip install \
        "jax[cuda12]==0.10.1" \
        "jaxlib==0.10.1"

# Create a non-root user matching the host user
ARG USERNAME=user
ARG UID=1000
ARG GID=1000
RUN userdel ubuntu 2>/dev/null || true && \
    groupdel ubuntu 2>/dev/null || true && \
    groupadd --gid ${GID} ${USERNAME} 2>/dev/null || true && \
    useradd --uid ${UID} --gid ${GID} --create-home ${USERNAME} && \
    chown -R ${UID}:${GID} /workspace /opt/plasmax

USER ${USERNAME}

CMD ["python3"]
