# Astron_1221_Project_2
# **Description**
#### **JWST Image Gallery Organizer**
Create a browsable catalog of JWST observations with metadata organization and simple image visualization. The challenge is working with astronomical FITS files and extracting their metadata into organized Pandas DataFrames.

Astronomy Context: The James Webb Space Telescope (JWST), launched in 2021, is the most powerful space telescope ever built, observing in infrared. FITS (Flexible Image Transport System) is the standard astronomical image format—it contains both image data and extensive metadata in "headers." JWST images have multiple extensions (science image, error map, data quality flags). You'll focus on extracting metadata from headers and organizing it, with optional image display. The "_i2d.fits" files are the processed, calibrated images ready for science.

Download 10-20 JWST FITS images from MAST Archive (use the web interface to search by target name and download "_i2d.fits" products—these are the final calibrated images). Use Astropy.io.fits to open files and read headers (this is taught in class), extract target info (RA/Dec from header keywords, filter name, exposure time, observation date), and optionally load image arrays. Organize metadata in Pandas DataFrame with columns for target, instrument, filter, observation date. Create catalog with filtering by target type, sort by date, and generate observing summaries. Use Matplotlib to display images if desired (optional but recommended).

The minimal version successfully downloads 10+ JWST FITS files, extracts metadata into organized DataFrame, implements filtering/sorting by multiple criteria, and creates a browsable catalog table. Advanced versions might extract full image arrays and display them, create color composites from multiple filters (RGB images), analyze image quality metrics from headers, build thumbnail gallery with subplots, or create observation statistics (most-observed targets, filter usage patterns, exposure time distributions).
# **Outline**
#### **Download MIRI Images from JWST Archive**
Choose which filters we want to download ie: F560W F1500W and F2550W
#### **Everything else**

