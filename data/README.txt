================================================================================
            PANORAMA IMAGE STITCHING — Dataset & Project Description
================================================================================

Team        :  FORtidata
Members     :  Aouami Salma
               Arrami Zineb
               Arbaoui Hiba
               Ayyadi Marwa

Course      :  Computer Vision
Notebook    :  code/stitching.ipynb

================================================================================
1. PROJECT OVERVIEW
================================================================================

This project implements a complete image stitching pipeline that combines two
or more overlapping photographs into a seamless wide-field panorama. The
approach is built entirely on classical computer vision techniques: keypoint
detection, local descriptor extraction, robust feature matching, homography
estimation, projective warping, and multi-scale image blending.

No neural network or learning-based component is used at any stage. All
geometric reasoning is handled through the mathematical formulations covered
in the course.

================================================================================
2. PIPELINE DESCRIPTION
================================================================================

The pipeline processes images sequentially (left to right) and consists of
seven main stages:

────────────────────────────────────────────────────────────────────────────────
Stage 1 — Preprocessing
────────────────────────────────────────────────────────────────────────────────
Each image is first resized so its largest dimension does not exceed 1500 px
(preserving the aspect ratio), making computation tractable without sacrificing
alignment accuracy.

CLAHE (Contrast Limited Adaptive Histogram Equalisation) is then applied in
the L channel of the LAB colour space. This locally amplifies contrast in
flat, underexposed regions — such as overcast sky or calm water — without
globally shifting colour balance. It significantly increases the number of
reliable keypoints detected in those otherwise featureless zones.

────────────────────────────────────────────────────────────────────────────────
Stage 2 — Keypoint Detection (SIFT)
────────────────────────────────────────────────────────────────────────────────
We use SIFT (Scale-Invariant Feature Transform) to detect keypoints and compute
descriptors. SIFT constructs a Difference-of-Gaussians (DoG) scale-space
pyramid and locates extrema that are stable across scale and rotation. Each
keypoint is assigned a dominant orientation from a local gradient histogram,
making the detector invariant to rotation and scale changes between frames.

Each keypoint is described by a 128-dimensional vector of gradient orientation
histograms computed over a 16×16 pixel neighbourhood divided into 4×4 cells.
The descriptor is L2-normalised and partially invariant to illumination change.

Parameters chosen: up to 5 000–8 000 keypoints per image with a lowered
contrast threshold (0.01–0.02 vs. the default 0.04) to ensure sufficient
detection in low-texture regions (sea surface, sky).

────────────────────────────────────────────────────────────────────────────────
Stage 3 — Feature Matching (FLANN + Lowe Ratio Test)
────────────────────────────────────────────────────────────────────────────────
Descriptors from each consecutive image pair are matched using FLANN
(Fast Library for Approximate Nearest Neighbours) with randomised kd-trees.
For each descriptor in image i, the two nearest neighbours in image i-1 are
retrieved (k=2 KNN search).

Lowe's ratio test is then applied: a match is accepted only if the distance to
the best neighbour is less than 75% of the distance to the second-best
neighbour. This rejects ambiguous matches where two database descriptors are
similarly close to the query, drastically reducing the number of false matches
passed to the homography estimator.

────────────────────────────────────────────────────────────────────────────────
Stage 4 — Homography Estimation (RANSAC)
────────────────────────────────────────────────────────────────────────────────
A projective homography H maps every point in image i to its corresponding
location in image i-1 via:

        λ [x', y', 1]ᵀ = H · [x, y, 1]ᵀ

H is a 3×3 matrix with 8 degrees of freedom estimated from point correspondences
using the Direct Linear Transform (DLT) algorithm.

Because the match set still contains outliers after the ratio test, we use
RANSAC (Random Sample Consensus) to robustly estimate H:
  - Randomly sample 4 correspondences and compute a candidate H.
  - Count inliers: matches whose symmetric reprojection error is below ε = 5 px.
  - Repeat for 10 000 iterations (confidence = 99.9%).
  - Re-fit H on all inliers of the largest consensus set.

For N images, pairwise homographies H_{i→i-1} are chained into a global
transform that maps every image to the coordinate system of image 0 (the
canvas reference):
        H_i^canvas = H_{i-1}^canvas · H_{i→i-1}

────────────────────────────────────────────────────────────────────────────────
Stage 5 — Image Warping
────────────────────────────────────────────────────────────────────────────────
Each image is warped onto a shared output canvas using cv2.warpPerspective,
which applies the chained homography with bilinear interpolation. The canvas
dimensions are computed by projecting the four corners of every image through
its global homography and taking the bounding box of all projected corners.
A translation offset is added so that no warped image falls outside the
positive-coordinate region of the canvas.

────────────────────────────────────────────────────────────────────────────────
Stage 6 — Blending
────────────────────────────────────────────────────────────────────────────────
Two blending strategies are implemented and compared:

(a) Distance-weighted alpha blending
    Each pixel's weight is its Euclidean distance to the nearest boundary of
    the contributing image (computed via a distance transform). Interior pixels
    receive higher weight than seam-adjacent ones. The final colour at each
    canvas pixel is the weighted average over all contributing images:

        I_blend(x,y) = Σ w_i(x,y)·I_i(x,y) / Σ w_i(x,y)

    This produces a smooth transition but can leave ghosting artefacts if the
    homography is not perfectly accurate.

(b) Multi-band blending (Laplacian pyramid)
    The image and its weight mask are both decomposed into a 5-level Laplacian
    pyramid. At each pyramid level k, the band-pass content (detail at scale 2^k)
    is blended using the correspondingly smoothed weight mask. High-frequency
    bands (sharp edges) are composited with a narrow transition zone; low-
    frequency bands (colour, large-scale shading) are blended over a wide zone.
    This eliminates ghosting at sharp boundaries while gracefully handling
    exposure or colour differences across the seam.

================================================================================
3. DATASETS
================================================================================

Three sets of real overlapping photographs are provided, processed in ascending
order of image count.

────────────────────────────────────────────────────────────────────────────────
data3/  —  Coastal Sunset  (2 images)
────────────────────────────────────────────────────────────────────────────────
  IMG_6625.jpg   Left panel  — rocky coastline, warm orange tones, surf visible
  IMG_6626.jpg   Right panel — open sea, cooler tones, distant headland

  Scene    : Seaside location photographed at dusk; camera panned horizontally.
  Overlap  : ~25–30% on the right side of IMG_6625 / left side of IMG_6626.
  Challenge: Large uniform sky and sea surface — low texture, sparse keypoints.

  Results  :
    SIFT keypoints / image   5 000  (contrast threshold 0.02)
    Matches (ratio test)     505
    RANSAC inliers           480 / 505  (95.0%)
    Reprojection error       0.15 px
    Output canvas            2 324 × 1 324 px

────────────────────────────────────────────────────────────────────────────────
data1/  —  Elevated Coastal Panorama  (3 images)
────────────────────────────────────────────────────────────────────────────────
  IMG_6622.PNG   Left panel
  IMG_6623.PNG   Centre panel
  IMG_6624.PNG   Right panel

  Scene    : Elevated viewpoint over a coastal landscape at golden hour;
             fields, cliff edge, and sea visible. Portrait-orientation shots.
  Overlap  : ~30% between consecutive frames.
  Challenge: Portrait orientation limits horizontal field of view per frame;
             slight perspective shift between hand-held shots.

  Results  :
    SIFT keypoints / image     5 000
    Pair 6622 ↔ 6623 — inliers : 232 / 281  (82.6%) — error: 1.86 px
    Pair 6623 ↔ 6624 — inliers : 201 / 226  (88.9%) — error: 1.60 px
    Output canvas              1 571 × 1 853 px

────────────────────────────────────────────────────────────────────────────────
data2/  —  Lake Annecy  (3 images)
────────────────────────────────────────────────────────────────────────────────
  IMG_6628.PNG   Left panel
  IMG_6629.PNG   Centre panel
  IMG_6630.PNG   Right panel

  Location : Esplanade des Marquisats, Lac d'Annecy, France
  Date     : 1 February 2025, 5:48 PM
  Scene    : Lake surface with sailing buoys, Alpine mountains behind, dock
             and moored boat visible in the rightmost frame.
  Overlap  : ~30–40% between consecutive frames.
  Challenge: Flat lake surface and overcast sky yield very sparse keypoints;
             required higher feature count (8 000) and lower threshold (0.01).

  Results  :
    SIFT keypoints / image     8 000  (contrast threshold 0.01)
    Pair 6628 ↔ 6629 — inliers : 266 / 353  (75.4%) — error: 1.08 px
    Pair 6629 ↔ 6630 — inliers : 1467 / 1511 (97.1%) — error: 0.89 px
    Output canvas              1 579 × 1 854 px

================================================================================
4. OUTPUT FILES  (generated by running the notebook)
================================================================================

  data/data3_panorama_alpha.png       data3 — distance-weighted alpha blend
  data/data3_panorama_multiband.png   data3 — multi-band Laplacian blend
  data/data1_panorama_alpha.png
  data/data1_panorama_multiband.png
  data/data2_panorama_alpha.png
  data/data2_panorama_multiband.png



================================================================================
