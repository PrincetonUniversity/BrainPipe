# BrainPipe: Registration and object detection pipelines for three-dimensional whole-brain analysis.

![plot](./static/SW_GrAbstr_STAR_1n.png)

BrainPipe [Pisano et al. 2021](https://www.cell.com/cell-reports/pdf/S2211-1247(21)01170-0.pdf>) is an automated transsynaptic tracing pipeline that combines anatomical assignment using volumetric registration and injection site segmentation with convolutional neural networks for cell detection.

Light-sheet microscopy to visualize whole-brain transsynaptic tracing:

![](./static/ClearingVideo_med.gif)

Automatic alignment of brain volumes to the Princeton Mouse Atlas for anatomical assignment:

![](./static/registration_med.gif)

## Installation
Please see [INSTALLATION.md](INSTALLATION.md) for installation instructions.

## Example use cases and tutorials
Please see [EXAMPLES.md](EXAMPLES.md) for basic BrainPipe use cases.

Demonstration datasets to ensure proper usage of BrainPipe and trained CNNs for H129 and PRV detection can be found [here](https://lightsheetatlas.pni.princeton.edu/public/brainpipe_demo_datasets/).

For CNN Demo: [EXAMPLES: CNN Demo](EXAMPLES.md#cnn-demo) Note, trained CNN's are a good starting point for transfer learning, but probably will not work out-of-the-box for other projects' datasets. For a deeper-dive into the BrainPipe pipeline, please see [IMPORTANT_FILES.md](IMPORTANT_FILES.md) for details. When starting to use your own data, BrainPipe expects certain formatting of images (see [FILE-FORMATTING.md](FILE-FORMATTING.md) for details).

![](./static/CNN_med.gif)

## Princeton Mouse Atlas

![](./static/PMA.gif)

The Princeton Mouse Atlas (PMA) is a light-sheet generated volumetric mouse atlas that is compatible with the Allen Brain Atlas. PMA data can be found [here](https://brainmaps.princeton.edu/2020/09/princeton-mouse-brain-atlas-links/).

## Aligned viral tracing injection data
![plot](/static/neuroglancer_injectionsite.png)
Aligned anterograde (HSV-H129) and retrograde (PRV-Bartha) viral tracing injection data from [Pisano et al. 2021](https://www.cell.com/cell-reports/pdf/S2211-1247(21)01170-0.pdf>) has been deposited [here](https://brainmaps.princeton.edu/2021/05/pisano_viral_tracing_injections/).
