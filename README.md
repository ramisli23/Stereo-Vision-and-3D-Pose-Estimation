# Stereo Vision and 3D Pose Estimation

A Computer Vision project focused on camera geometry, stereo vision, planar pose estimation, and 3D reconstruction using Python and NumPy.

The project explores the mathematical foundations of geometric computer vision through practical experiments, from epipolar geometry and fundamental matrix estimation to pose refinement and 3D cube projection.

## Overview

This project implements and evaluates several computer vision techniques for recovering geometric relationships between cameras and estimating the 3D pose of objects from image observations.

The experiments cover two main areas:

- **Stereo Vision and Epipolar Geometry:** Understanding the geometric relationship between two camera views and estimating the fundamental matrix.
- **3D Pose Estimation:** Recovering camera pose from planar points, refining the estimation, and studying planar pose ambiguity.

## Implemented Methods

- Stereo vision and epipolar geometry
- Fundamental matrix estimation using the normalized eight-point algorithm
- Singular Value Decomposition (SVD)
- Direct Linear Transform (DLT)
- Perspective-n-Point (PnP) pose estimation
- Homography-based pose estimation
- Levenberg–Marquardt (LM) optimization
- Reprojection error evaluation
- 3D cube projection
- Planar pose ambiguity analysis

## Experimental Results

The implementation successfully validated all TP3 steps (Workshops 1 and 2).

### Pose Estimation and Optimization

| Metric | Before LM | After LM |
|---|---:|---:|
| Reprojection RMS | 2.653 px | 0.704 px |
| Rotation error | 6.39° | 0.56° |

The Levenberg–Marquardt refinement significantly improved the estimated camera pose and reduced the reprojection error.

### 3D Cube Projection

| Metric | Result |
|---|---:|
| Mean deviation from reference cube | 1.02 px |
| Maximum deviation | 1.85 px |

### Planar Pose Ambiguity

The experiment identified two plausible solutions for a planar marker of 60 mm at a distance of 1000 mm.

| Solution | Reprojection RMS | Rotation Error |
|---|---:|---:|
| Minimum 1 | 0.303 px | 1.6° |
| Minimum 2 | 0.527 px | 46.7° |

These results demonstrate that different 3D poses can produce similar image projections when using planar observations.

## Project Structure

```text
Stereo-Vision-and-3D-Pose-Estimation/
│
├── annexe_TP3_squelette.py
├── annexe_TP3_donnees.json
├── cube.png
├── README.md
├── requirements.txt
│
├── results/
│   ├── etapes_1_6.png
│   ├── validation_finale.png
│   └── cube_projete.png
│
└── report/
    └── rapport_TP3.pdf
```

*The `results/` and `report/` directories are optional and should be added when their files are available.*

## Technologies Used

- **Python** – Main programming language
- **NumPy** – Numerical computations and matrix operations
- **Matplotlib** – Data visualization and geometric projections
- **SciPy** – Numerical optimization, if used by the implementation

## Installation

Clone the repository:

```bash
git clone https://github.com/ramisli23/Stereo-Vision-and-3D-Pose-Estimation.git
cd Stereo-Vision-and-3D-Pose-Estimation
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

Execute the main Python script:

```bash
python annexe_TP3_squelette.py
```

The script runs the experiments and displays the validation results in the terminal.

## Key Findings

- Geometric constraints are essential for estimating camera relationships.
- The normalized eight-point algorithm provides a practical approach to fundamental matrix estimation.
- Pose refinement using Levenberg–Marquardt can significantly reduce reprojection error.
- Planar pose estimation may produce multiple plausible solutions.
- Reprojection error is an important metric for evaluating geometric accuracy.

## Author

**Rami Slimani**

Artificial Intelligence Engineering Student  
Badji Mokhtar University – Annaba, Algeria

- GitHub: [@ramisli23](https://github.com/ramisli23)

## Acknowledgments

This project was developed as part of the Computer Vision practical work (TP3), covering Workshops 1 and 2.

---

*Exploring the geometry behind computer vision, one projection at a time.*