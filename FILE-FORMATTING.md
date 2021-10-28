# File Formatting
Different light-sheet microscopes output files in different formats. BrainPipe is set up to detect files of a specific format using regular expressions (regex). There are a few options here:  

1. Re-format your raw light-sheet files to the default format that we use in [EXAMPLES](EXAMPLES.md), i.e.:
	- Raw files must be in a single directory and must be stored in 1 file per Z plane. The plane can be any orthogonal cut through the data (e.g. sagittal, horizontal or coronal). 
	- Files must have the format:
```
raw_RawDataStack[{X_TILE} x {Y_TILE}]_C{LIGHTSHEET_LR_CODE}_xyz-Table Z{ZPLANE_CODE}_UltraII Filter{FILTER_CODE}.ome.tif
```  
where {X_TILE} and {Y_TILE} are 2 digit 0-padded strings representing the row and column of the tile in the grid, {LIGHTSHEET_LR_CODE} is a 2 digit 0-padded string representing the left or right lightsheet ("00" = Left, "01" = Right), {ZPLANE_CODE} is a 4 digit 0-padded string representing the z plane index, and {FILTER_CODE} is a 4 digit 0-padded string representing the index of the laser wavelength used in this directory. All of these quantities are 0-indexed.

Here is an example to illustrate each of the variables:
```
raw_RawDataStack[01 x 02]_C00_xyz-Table Z0001_UltraII Filter0001.ome.tif
```
This is the second Z plane (Z0001) from the second row and third column (01 x 02) from the left lightsheet (C00) from the second filter used in this directory.  Note that if you only have files from a single lightsheet use C00 for the LIGHTSHEET_LR_CODE. Similarly, if you only have one filter use 0000 for the filter code. And use 00 x 00 for the {X_TILE} x {Y_TILE} if you don't have tiled data.  

2. If you do not wish to reformat your files, then create your own regex string and add it to the top of `lst` in `regex_determiner()` in `tools/imageprocessing/preprocessing.py`. Note that the regex needs to be able to match all of the same variables as the default regex. If your files only ever use one lightsheet code, you can set the regex so that match is always blank. This may take some trial and error. A good test to see if your regex worked is if step 0 runs without error and the output is sensible, e.g.:
```
*******************STEP 0**********************************

25 *Complete* Zplanes found for /jukebox/LightSheetData/lavision_testdata/4x_example/lavision_4x_cellch_z4um_10percentoverlap_demodataset

Checking for bad missing files:
     Bad planes per channel:
     {'0000': []}

1 Channels found

3x by 3y tile scan determined

2 Light Sheet(s) found. 1 Horizontal Focus Determined
```