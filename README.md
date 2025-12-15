# Foundations in AI – Multimodal Sensor Fusion (KITTI)

This repository contains the implementation for a multimodal perception pipeline
developed for the **Foundations in AI** coursework.

The system fuses **RGB camera images, LiDAR point clouds, and GPS/IMU data**
from the KITTI dataset to analyse scene structure and visual busyness.

---

## Features
- YOLOv5-based object detection (semantic perception)
- Camera–LiDAR projection and fusion
- Scene busyness metrics (2D occupancy, LiDAR density, entropy)
- Geometric and semantic scene analysis

---

## Dataset
This project uses the **KITTI Raw Dataset**:

- `2011_10_03_drive_0047_sync`

Due to licensing and size constraints, the dataset is **not included** in this repository.

---

## Installation

```bash
pip install -r requirements.txt
