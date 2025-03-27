# 📸 Say Cheese! - Face Recognition Attendance System

A modern, interactive attendance tracking system that uses facial recognition to identify and record attendance automatically. The system features a playful, sketchy UI design that makes the attendance process fun rather than tedious.

![Attendance System Banner](https://example.com/banner-image.png)

## Table of Contents
- [📸 Say Cheese! - Face Recognition Attendance System](#-say-cheese---face-recognition-attendance-system)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Key Features](#key-features)
  - [Technical Architecture](#technical-architecture)
  - [Face Recognition Pipeline](#face-recognition-pipeline)
  - [Database Structure](#database-structure)
    - [Users Table](#users-table)
    - [Embeddings Table](#embeddings-table)
    - [Attendance Table](#attendance-table)
  - [User Registration Process](#user-registration-process)
  - [Attendance Tracking Process](#attendance-tracking-process)
  - [Models and Algorithms](#models-and-algorithms)
    - [MTCNN (Multi-task Cascaded Convolutional Networks)](#mtcnn-multi-task-cascaded-convolutional-networks)
    - [FaceNet (InceptionResnetV1)](#facenet-inceptionresnetv1)
    - [Similarity Measurement](#similarity-measurement)
  - [Performance Optimizations](#performance-optimizations)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Dependencies](#dependencies)

## Overview

Say Cheese! is a fully-featured face recognition attendance system designed to simplify attendance tracking in educational institutions, workplaces, or events. The system replaces traditional roll calls with an automated, engaging process that identifies individuals through their facial features and logs their attendance in real-time.

## Key Features

- **Facial Recognition Based Attendance**: Automatically identifies individuals and records attendance
- **User-Friendly Interface**: Intuitive, sketchy UI design with fun animations and interactions
- **Real-Time Tracking**: Displays attendance records immediately after recognition
- **Multiple Registration Methods**: Support for both camera capture and photo uploads
- **User Management**: Add, view, and remove users with an easy-to-use interface
- **Secure Database**: Stores face embeddings rather than actual face images for privacy
- **Responsive Design**: Works on various screen sizes and devices
- **Fun Interactive Elements**: Countdown timers, progress indicators, and encouraging messages

## Technical Architecture

The system is built with the following components:

- **Backend**: Python Flask web application
- **Database**: SQLite with proper indexing and foreign key constraints
- **Face Detection**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Face Recognition**: FaceNet (Inception-ResNet-V1 architecture)
- **Frontend**: HTML5, CSS3, JavaScript with a sketchy, playful design

## Face Recognition Pipeline

The face recognition process follows these steps:

1. **Image Acquisition**: Capture frames from the camera or process uploaded images
2. **Face Detection**: MTCNN detects and localizes faces in the image
3. **Face Alignment**: Faces are aligned based on detected facial landmarks
4. **Feature Extraction**: The aligned face is passed through FaceNet to generate a 512-dimensional embedding
5. **Similarity Matching**: The generated embedding is compared with stored embeddings using cosine similarity
6. **Identity Verification**: The identity with the highest similarity score above a threshold (0.6) is considered a match

## Database Structure

The system uses a SQLite database with three main tables:

### Users Table
- `id`: Primary key
- `name`: User's full name
- `user_id`: Unique identifier for the user
- `registration_date`: Date when the user was registered

### Embeddings Table
- `id`: Primary key
- `user_id`: Foreign key referencing Users table
- `embedding`: Binary blob storing the 512-dimensional face embedding vector

### Attendance Table
- `id`: Primary key
- `user_id`: Foreign key referencing Users table
- `name`: User's name (denormalized for performance)
- `date`: Date of attendance
- `time`: Time of attendance

## User Registration Process

The registration process follows these steps:

1. **User Information Entry**: Collect name and ID
2. **Face Data Collection**: Capture 10 different facial images through:
   - Live camera capture with visual guidance
   - Photo uploads
3. **Face Processing**: 
   - MTCNN detects faces in each image
   - FaceNet extracts embedding vectors for each face
4. **Embedding Aggregation**: 
   - Average all 10 embeddings to create a robust representation
   - This helps account for variations in lighting, expression, and angle
5. **Database Storage**: 
   - Store user information in the Users table
   - Store the final embedding in the Embeddings table

## Attendance Tracking Process

The attendance tracking process works as follows:

1. **Camera Activation**: System activates the camera when in attendance mode
2. **Frame Processing**: 
   - Capture frames from the camera at regular intervals
   - For performance, only every 3rd frame is processed
3. **Face Detection**: 
   - MTCNN detects faces in the frame
   - Bounding boxes are drawn around detected faces
4. **Identity Determination**:
   - Extract face embeddings using FaceNet
   - Compare with database embeddings using cosine similarity
   - Identify the person with the highest similarity score above threshold
5. **Attendance Recording**:
   - Check if attendance is already recorded for today
   - If not, add a new record to the Attendance table
   - Update the attendance display in real-time
   - Show a notification acknowledging the person's attendance

## Models and Algorithms

### MTCNN (Multi-task Cascaded Convolutional Networks)
- **Purpose**: Face detection and facial landmark localization
- **Architecture**: Cascade of CNNs (P-Net, R-Net, O-Net)
- **Configuration**: 
  - `image_size`: 160
  - `margin`: 20
  - `min_face_size`: 20
  - `thresholds`: [0.5, 0.6, 0.6] (lower thresholds for easier detection)
  - `factor`: 0.709
  - `select_largest`: False (don't just select the largest face)
  - `keep_all`: True (detect multiple faces if present)

### FaceNet (InceptionResnetV1)
- **Purpose**: Generate face embeddings
- **Architecture**: Inception-ResNet-V1
- **Pre-training**: Trained on VGGFace2 dataset
- **Output**: 512-dimensional embedding vector representing facial features
- **Performance**: Runs in evaluation mode for optimal inference
- **Hardware Acceleration**: Automatically uses CUDA if available

### Similarity Measurement
- **Method**: Cosine Similarity
- **Threshold**: 0.6 (calibrated for optimal recognition accuracy)
- **Advantages**: Scale-invariant and efficient for high-dimensional vectors

## Performance Optimizations

The system includes several optimizations:

1. **Frame Skipping**: Only processes every 3rd frame for better performance
2. **Resolution Adjustment**: Camera resolution is set to 640x480 for optimal balance
3. **Database Connection Management**: Uses context managers to prevent connection leaks
4. **Retry Mechanism**: Implements exponential backoff for database lock issues
5. **Fallback Methods**: CSV backup for attendance records in case of database errors
6. **Hardware Acceleration**: Automatically uses GPU when available for neural network operations

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create necessary directories:
   ```bash
   mkdir -p Attendance static/faces static/embeddings static/images static/temp_embeddings
   ```
4. Run the application:
   ```bash
   python app.py
   ```

## Usage

1. **Add Users**: Click "Add New Face" and follow the registration process
2. **Take Attendance**: Click "Smile for Attendance" to start the camera
3. **View Attendance**: Current day's attendance is shown in the main interface
4. **Manage Users**: Click "Manage Users" to view or remove registered users

## Dependencies

- Flask
- OpenCV
- PyTorch
- facenet-pytorch
- NumPy
- Pandas
- SQLite3
- PIL (Pillow)

