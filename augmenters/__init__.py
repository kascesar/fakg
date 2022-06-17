'''
FAKG AUGMENTERS:

This module contains all "augmenter" and "ImagePipe" to contruct a pipe line to
apply augmentation techniques.

the suported augmenters are listed bellow
 ____________________________________
| Estructural Augmenters (image data)|
+------------------------------------+
  1. Vflip (vertical image flip)
  2. Hflop (Horizontal image flip)
  3. RandomRotation (Randomly rotate image)
  4. RandomShear (Randomly shear image in both axes)
  5. RandomShift (Randomly shift image in x,y direction)
  5. RandomCrop (Random crop an image, like RandomShift,
                 this is more like "zoom in img")
  6. CutOut (randomly cut porcentage of image)
 ______________________________________________
| Color&Noise&filtering Augmenters (image data)|
+----------------------------------------------+

  1. KMeansCC (Kmeans Color clustering)
  2. BrightnessJitter (Random brigthness Jitter)
  3. SaturationJitter (Random saturatin jitter)
  4. ContrastJitter ()
  5. GaussianFlitering (Gaussian filtering)

 _________________
| pipe augmenters |
+-----------------+

1. ImagePipe
-------------------------------------------------------------------------------
 ____
|TODO|
+----+

--NOISE AUGMENTERS--
    * noises salt-pepper
    * noise Gausian

--FILTERING AUGMENTERS--
    * blur(Mean)
    * Median


--ImagePipe---
    * rebuild ImagePipe for tree like path (like imau)
    * decorator to test if an augmenter return no-boxes when it recibe boxes

--ImageLayer--
    * Rebuild ImageLayer for write less code for every augmenter

'''

from fakg.augmenters.image import (
                                   # Estructural augmenters
                                   Vflip,           # 1
                                   Hflip,           # 2
                                   RandomRotation,  # 3
                                   RandomShear,     # 4
                                   RandomShift,     # 5
                                   RandomCrop,      # 6
                                   CutOut,          # 7
                                   # Color&Noise&filtering Augmenters
                                   KMeansCC,           # 1
                                   # BrightnessJitter,   # 2  BUUUUGED
                                   SaturationJitter,   # 3
                                   # ContrastJitter,     # 4 HAS A BUGGGGGG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                   GaussianFlitering,  # 5
                                   )

from fakg.augmenters.pipe import ImagePipe

__all__ = [
           'Vflip',
           'Hflip',
           'RandomRotation',
           'RandomShear',
           'RandomShift',
           'RandomCrop',
           'CutOut',

           'KMeansCC',
           'BrightnessJitter',
           'SaturationJitter',
           #'ContrastJitter',
           'GaussianFlitering',
           'ImagePipe',
           ]
