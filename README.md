# compare-images
 Implementation of nine evaluation metrics to access the similarity between two images and obtain the regions of the two input images that differ. The nine metrics are as follows:
 * <i>Root mean square error (RMSE)</i>,
 * <i>Peak signal-to-noise ratio (PSNR)</i>,
 * <i>Structural Similarity Index (SSIM)</i>,
 * <i>Feature-based similarity index (FSIM)</i>,
 * <i>Information theoretic-based Statistic Similarity Measure (ISSM)</i>,
 * <i>Signal to reconstruction error ratio (SRE)</i>,
 * <i>Spectral angle mapper (SAM)</i>,
 * <i>Universal image quality index (UIQ)</i>,
 * <i>Visual Information Fidelity (VIFP)</i>,
 
 ## Instructions
 The following step-by-step instructions will guide you through installing this package and run evaluation using the command line tool.

 **Note:** Supported python versions are 3.6, 3.7, 3.8, and 3.9.
 
 ### Install package library
 
```bash
pip install image-similarity-measures
```

```bash
python3 -m pip install -r requirements.txt
```

### Usage

#### Parameters
```
  --org_img_path FILE_PATH   Path to original input image
  --pred_img_path FILE_PATH  Path to predicted image
  --metric METRIC       select an evaluation metric (fsim, issm, psnr, rmse,
                        sam, sre, ssim, uiq, vifp, all) (can be repeated)
```
 
 #### Terminal
```bash
python main.py --org_img_path FILE_PATH --pred_img_path FILE_PATH --metric METRIC
```
#### Example
 
```bash
python main.py --org_img_path Images/1.png --pred_img_path Images/2.png --metric all
```

## References

<strong>Müller, M. U., Ekhtiari, N., Almeida, R. M., and Rieke, C.: SUPER-RESOLUTION OF MULTISPECTRAL
SATELLITE IMAGES USING CONVOLUTIONAL NEURAL NETWORKS, ISPRS Ann. Photogramm. Remote Sens.
Spatial Inf. Sci., V-1-2020, 33–40, https://doi.org/10.5194/isprs-annals-V-1-2020-33-2020, 2020.</strong>

H. R. Sheikh and A. C. Bovik, “Image information and visual quality,” Image Processing, IEEE Transactions on, vol. 15, no. 2, pp. 430–444, 2006.

V. Baroncini, L. Capodiferro, E. D. Di Claudio, and G. Jacovitti, “The polar edge coherence: a quasi blind metric for video quality assessment,” EUSIPCO 2009, Glasgow, pp. 564–568, 2009.

Z. Wang, E. P. Simoncelli, and A. C. Bovik, “Multiscale structural similarity for image quality assessment,” Conference Record of the Thirty-Seventh Asilomar Conference on Signals, Systems and Computers, 2003, vol. 2, pp. 1398–1402.

Mittal, Anish, Rajiv Soundararajan, and Alan C. Bovik. "Making a completely blind image quality analyzer." Signal Processing Letters, IEEE 20.3 (2013): 209-212.

<i><a href="https://github.com/aizvorski/video-quality">Video Quality Metrics - aizvorski</a></i>

<i><a href="https://github.com/up42/image-similarity-measures">image-similarity-measures - up42</a></i>
