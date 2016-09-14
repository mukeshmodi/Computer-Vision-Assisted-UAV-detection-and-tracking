obj=VideoReader('Flying_object.avi');
videoplayer=vision.VideoFileReader();
vidFrames=read(obj);
numFrames=get(obj,'numberOfFrames');% 
[c r t n]=size(vidFrames);
bw=zeros(c,r);
kalmanFilter = []; isTrackInitialized = false;
blobAnalyzer = vision.BlobAnalysis('AreaOutputPort',false,'MinimumBlobArea',20);

for k=1:100
im=vidFrames(:,:,:,k);

k
for i=1:c %loop to read all pixels
    for j=1:r
        
        if(im(i,j,1)>160 && im(i,j,2)>190 && im(i,j,3)>170)
        bw(i,j)=1;
        end
    end
   if k==2
   imwrite(im,'im.bmp');
    end
   
end
 bw=bwareaopen(bw,90);% removing noise
    cc=bwconncomp(bw);
    
   
    img=logical(bw);
   detectedLocation= step(blobAnalyzer,img);
    
   isObjectDetected = size(detectedLocation, 1) > 0;
   if ~isTrackInitialized
         if isObjectDetected
         kalmanFilter = configureKalmanFilter('ConstantAcceleration',detectedLocation(1,:), [1 1 1]*1e5, [25, 10, 10], 25);
             isTrackInitialized = true;
         end
        label = ''; circle = zeros(0,500);
   else
         if isObjectDetected
         predict(kalmanFilter);
         trackedLocation = correct(kalmanFilter, detectedLocation(1,:));
         label = 'Corrected';
       
         end
          circle = [trackedLocation, 5];
   end
   
   
   mask = imopen(img, strel('rectangle', [3,3]));
   
  mask = imclose(mask, strel('rectangle', [15, 15]));
        mask = imfill(mask, 'holes');
   
     [ centroids, bboxes] = step(blobAnalyzer,mask);
    
     
     
     
     bboxes
     
     
frame = insertObjectAnnotation(im, 'rectangle', ...
                    bboxes, 'UAV Detected');
    imshow(frame);
    
   bw =zeros(c,r);
   pause(0.05);
    
end







