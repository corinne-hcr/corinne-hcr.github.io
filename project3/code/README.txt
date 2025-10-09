CS180 Project 3 Part A - Image Warping and Mosaicing
=====================================================

OVERVIEW
--------
Create panoramic images by computing homographies, warping images, and 
blending them together.

FILES
-----
- main.ipynb: Implementation code
- index.html: Project webpage with results
- Images: b1.jpg, b2.jpg, phy1-3.jpg, a1.jpg, a2.jpg
- Correspondence points: *.json files

RESULTS (see index.html for details)
------------------------------------

A.1 Shoot the Pictures
- 3 image sets captured with fixed COP, camera rotation only
- Images resized to 800px width

A.2 Recover Homographies  
- Implemented computeH(im1_pts, im2_pts) using SVD least squares
- Visualized correspondence points on image pairs
- Displayed system of equations (Ah = 0)
- Recovered and displayed H matrices for all image pairs

A.3 Warp the Images
- Implemented warpImageNearestNeighbor() and warpImageBilinear()
- Used inverse warping to avoid holes
- 3 rectification examples: projection screen, poster, door
- Compared nearest neighbor vs bilinear interpolation quality/speed

A.4 Blend Images into Mosaics
- 3 panoramas: Hallway 1, Physics Building, Hallway 2
- Used weighted averaging with linear alpha feathering
- One-shot warping approach with all images warped to reference frame
- Explained blending procedure in detail

See index.html for all visualizations and detailed explanations.