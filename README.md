# Semi-Pixelated to Perfect Pixel

This script converts semi-pixelated, grid-like images (like pixel art or upscaled sprites with visible grid lines) into clean, perfectly pixel-aligned images.

It detects the grid automatically, finds the center of each cell, samples the color at that point, and rebuilds a compact, crisp version of the image.

# Requirements:
Python 3.8+

# How to use
## Single Image Mode (default)
-Simply put your image at the same folder as the script, and name the image:

  -input_image.png
  
-Run the script

-The plots that appear show the steps that are being taken. Simply close the currect plot to progress to the next step. (You can disable showing the plots with changing SHOW_PLOTS = False. That way the conversion will happen without any further user input)

-fixed_image.png will be created when all steps are completed.

-DONE.
## Batch Mode
-Create two folders next to the script:

  -input_images/
  
  -output_images/
  
-Put all images you want to process into input_images/.

-Open the script and set:

  -BATCH_MODE = True
  
-Run the script.

-All the images in the input_folder will be converted, and the converted imaged will be placed in the output_folder. (plots are disabled in Batch mode)
## Parameters:
There are some parameters that can be tweaked. Depending on the image, slightly changing the parameters might help produce a better pixel perfect image.

The parameters are:

  CANNY_THRESHOLD_LOW = 30
  
  CANNY_THRESHOLD_HIGH = 100
  
  MIN_EDGE_PIXELS = 25
  
  DBSCAN_EPS = 3
  
  DBSCAN_MIN_SAMPLES = 1

# Notice 1
Whether the results are good or not depend on the characteristics of the input image. The process has a few flaws. For example, if an image has too few edges in some places, these won't be detected and the converted image might be distorted. Too many edges can also cause problems, in part because DBSCAN may merge them into one edge. Tweaking the parameters a bit can help, but not always.

# Notice 2
This script was made for fun in one day in Google Colab in 21 Jul 2023, and was then refined a bit to work as a standalone script (as well as a few other minor changes). I like pixel art, and I had this random idea about creating real, pixel perfect art from AI. And this was the result Hopefully this may give some ideas to other people!
