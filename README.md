# Messgerät

## Pipeline

- Read Config settings from CLI arguments
	- Threshold: Default 125
	- Image path: Required
	- Step in the pipline that gets output as image: Default last step
	- Reference Object, either 2€, 1€ or 50 cent (optional automate?)
- Load Image
- Optional: Make it work on other then white paper
- Convert to Binary (Adjustable Threshold)
- Detetct Circle (Hough Transform or Detect Konturs etc)
- Set Reference by Classifying Circle with 2Euro or 50Cent...
- Detect Regions of Objeckt that is being measured
- Create Bounding box