# Sample-Classification-Using-openCV

## Introduction:

The objective was to categorize samples as positive or negative based on color masking. When a sample (e.g., RT-qPCR or any biosample) undergoes testing, it is treated with a dye that changes its color to pink or yellow, indicating its positivity or negativity for a specific reagent. The imaging system, operated via an Arduino, captures images of the test tubes containing the samples. Subsequently, a script is employed to process these images immediately after capture. This script employs color masking techniques to isolate the portion of the test tube containing the sample. By analyzing the color intensity of this portion, a score is computed, indicating the degree of pinkness or yellowness. Based on predefined thresholds, the sample is then classified as positive or negative.

Now, let's analyze the code in detail:

## Supporting Functions:

Several supporting functions are defined within the `Test` function to facilitate image analysis. Here's a brief overview of each:

1. **BuildSpace:** This function builds a color space based on given colors.
2. **Mahanalobis:** Calculates the Mahalanobis distance between two vectors.
3. **FindTube:** Determines the tube number based on a given value.
4. **dist:** Calculates the distance between a value and a tuple.
5. **ValidContour:** Checks if a contour is valid based on its area.
6. **Score:** Computes scores based on detected colors.
7. **GetExpectedPixel:** Calculates the expected pixel value of a region in the image.
8. **Results:** Generates analysis results in the form of CSV files.

---

## Analysis Process:

The analysis process involves several steps:

1. **Image Preprocessing:** The image is cropped and filtered to enhance color detection.
2. **Color Detection:** Colors are detected within specific regions of interest, and contours are identified.
3. **Score Calculation:** Scores are computed based on the detected colors and their distances from predefined color spaces.
4. **Results Generation:** Analysis results, including scores for each sample, are generated and saved to CSV files.

---

## Conclusion:

The `Test` function provides a comprehensive solution for analyzing images, detecting specific colors, and generating analysis results. It utilizes various techniques such as color space definition, Mahalanobis distance calculation, contour detection, and score computation to achieve accurate analysis results.
