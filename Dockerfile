# Use official Python base image
FROM python:3.12.1

# Set environment variables for non-interactive apt installs and GUI display
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=host.docker.internal:0.0

# Install system packages (Qt, OpenCV dependencies, SQLite3, etc.)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    qtbase5-dev \
    qtbase5-dev-tools \
    qtchooser \
    qt5-qmake \
    libxcb-xinerama0 \
    libxcb-xinerama0-dev \
    libxcb1 \
    libx11-xcb1 \
    libxkbcommon-x11-0 \
    libglu1-mesa \
    x11-xserver-utils \
    cmake \
    build-essential \
    libboost-all-dev \
    libdlib-dev \
    ffmpeg \
    x11-apps \
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Optional: if your app writes to a DB file, create a persistent volume
VOLUME ["/app/db"]

# Copy dependency file and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Set Qt plugin path manually to avoid "xcb" plugin error
ENV QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms

# Copy the rest of the app
COPY . /app
WORKDIR /app

# Default command to run your main GUI app
CMD ["python", "main.py"]
