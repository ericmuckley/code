function [BW,maskedImage] = segmentImage(im)
%segmentImage segments image using auto-generated code from imageSegmenter App
%  [BW,MASKEDIMAGE] = segmentImage(IM) segments image IM using auto-generated
%  code from the imageSegmenter App. The final segmentation is returned in
%  BW and a masked image is returned in MASKEDIMAGE.

% Auto-generated by imageSegmenter app on 15-Mar-2016
%----------------------------------------------------


% Convert to grayscale
im = rgb2gray(im);

% Initialize segmentation with threshold
mask = im>25;

% Evolve segmentation
BW = activecontour(im, mask, 100, 'edge');

% Form masked image from input image and segmented image.
maskedImage = im;
maskedImage(~BW) = 0;
end
