imageCalculator("Subtract stack", "spatialJacobian.tif","spatialJacobian-1.tif");
imageCalculator("Multiply stack", "spatialJacobian.tif","PRA_mask.tif");
imageCalculator("Divide stack", "spatialJacobian.tif","PRA_mask.tif");
//run("Brightness/Contrast...");
run("phase");
