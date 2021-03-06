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
mask = im>50;

% Evolve segmentation
BW = activecontour(im, mask, 100, 'edge');

% Form masked image from input image and segmented image.
maskedImage = im;
maskedImage(~BW) = 0;
end


function batchsegmentImage(inDir, outDir)
%batchsegmentImage Batch process images using segmentImage
% batchsegmentImage(inDir, outDir) processes each image file in inDir using
% the function segmentImage and writes the resulting image to outDir.
%

% Auto-generated by imageBatchProcessor app on 15-Mar-2016
%----------------------------------------------------------

if(nargin<2)
    outDir = '\\Client\C$\Users\a6q\Desktop\CNT_images_segmentImage';
end
if(nargin<1)
    inDir = '\\Client\C$\Users\a6q\Desktop\CNT_images';
end

includeSubdirectories = true;

% All extensions that can be read by IMREAD
imreadFormats       = imformats;
supportedExtensions = [imreadFormats.ext];
% Add dicom extensions
supportedExtensions{end+1} = 'dcm';
supportedExtensions{end+1} = 'ima';
supportedExtensions = strcat('.',supportedExtensions);
% Allow the 'no extension' specification of DICOM
supportedExtensions{end+1} = '';

% Create a image data store that can read all these files
imds = datastore(inDir,...
    'IncludeSubfolders', includeSubdirectories,...
    'Type','image',...
    'FileExtensions',supportedExtensions);
imds.ReadFcn = @readSupportedImage;

% Process each image using segmentImage
for imgInd = 1:numel(imds.Files)
    
    inImageFile  = imds.Files{imgInd};
    
    % Output has the same sub-directory structure and file extension as
    % input
    outImageFile = strrep(inImageFile, inDir, outDir);
    
    try
        % Read
        im = imds.readimage(imgInd);
        % Process
        im = segmentImage(im);
        
        % Create (sub)directory if needed
        outSubDir = fileparts(outImageFile);
        createDirectory(outSubDir);
        
        % Write
        if(isdicom(inImageFile))
            dicommeta = dicominfo(inImageFile);
            dicomwrite(im, outImageFile, dicommeta, 'CreateMode', 'copy');
        else
            imwrite(im, outImageFile);
        end
        disp(['PASSED:', inImageFile]);
        
    catch allExceptions
        disp(['FAILED:', inImageFile]);
        disp(getReport(allExceptions,'basic'));
    end
    
end
end


function img = readSupportedImage(imgFile)
% Image read function with DICOM support
if(isdicom(imgFile))
    img = dicomread(imgFile);
else
    img = imread(imgFile);
end
end

function createDirectory(dirname)
% Make output (sub) directory if needed
if exist(dirname, 'dir')
    return;
end
[success, message] = mkdir(dirname);
if ~success
    disp(['FAILED TO CREATE:', dirname]);
    disp(message);
end
end
