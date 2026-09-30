# Environment image: mount a clean checkout at /workspace (see docs/artifact.md).
FROM ghcr.io/astral-sh/uv:0.12.4 AS uv
FROM python:3.12-slim-trixie

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libnuma1 libssl-dev \
        pkg-config procps unzip \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/

# Match the host user so bind-mounted outputs do not become root-owned.
ARG VOSTI_UID=1000
ARG VOSTI_GID=1000
RUN groupadd --gid ${VOSTI_GID} vosti \
    && useradd --uid ${VOSTI_UID} --gid ${VOSTI_GID} --home-dir /opt/vosti-user --create-home vosti \
    && mkdir -p /opt/vosti-deps /opt/vosti-env /opt/verus /workspace \
    && chown -R vosti:vosti /opt/vosti-deps /opt/vosti-env /opt/verus /workspace
USER vosti
ENV PATH="/opt/vosti-env/bin:/opt/vosti-user/.cargo/bin:${PATH}" \
    UV_PROJECT_ENVIRONMENT=/opt/vosti-env \
    UV_PYTHON_DOWNLOADS=never \
    PYO3_PYTHON=/opt/vosti-env/bin/python \
    LD_LIBRARY_PATH=/usr/local/lib \
    VERUS=/opt/verus/verus-x86-linux/verus \
    CARGO_VERUS=/opt/verus/verus-x86-linux/cargo-verus \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /opt/vosti-deps
COPY --chown=vosti:vosti pyproject.toml uv.lock rust-toolchain.toml ./
RUN curl --fail --location --retry 3 https://sh.rustup.rs -o /tmp/vosti-rustup.sh \
    && sh /tmp/vosti-rustup.sh -y --no-modify-path --default-toolchain none \
    && rm /tmp/vosti-rustup.sh \
    && rustup toolchain install "$(python3 -c 'import tomllib; print(tomllib.load(open("rust-toolchain.toml", "rb"))["toolchain"]["channel"])')" \
    && rustup show active-toolchain

# Keep the complete Verus distribution, including its Z3 and libraries.
RUN curl --fail --location --retry 3 \
        https://github.com/verus-lang/verus/releases/download/release%2F0.2026.08.23.fbbbbcf/verus-0.2026.08.23.fbbbbcf-x86-linux.zip \
        -o /tmp/vosti-verus.zip \
    && echo 'b65483714e6bf2ae72bfe7c7199e1c608495a941306b979260e9ed585d5899c9  /tmp/vosti-verus.zip' | sha256sum --check \
    && unzip -q /tmp/vosti-verus.zip -d /opt/verus \
    && rm /tmp/vosti-verus.zip \
    && "$VERUS" --version
RUN uv sync --locked --no-cache --python /usr/local/bin/python3.12 \
    && python -c 'import torch, triton, z3; print(torch.__version__, triton.__version__, z3.get_version_string())'

WORKDIR /workspace
CMD ["bash"]
