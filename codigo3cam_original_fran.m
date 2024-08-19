% Detect Face
% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

% Crear un objeto VideoReader
videoReader = VideoReader('videoFrase_1_genesis1.mp4');
%videoReader = VideoReader('./Frase2/videoFrase_2_diana1.mp4');
numFrames = videoReader.NumFrames;

% Escoger el tiempo en segundos del cuadro deseado
%desiredTime = 1; % por ejemplo, el cuadro a los 10 segundos

% Establecer el tiempo actual del videoReader al tiempo deseado
%videoReader.CurrentTime = desiredTime;
videoFrame = readFrame(videoReader);

% Convertir el frame a escala de grises
grayFrame = rgb2gray(videoFrame);

% Detectar la cara usando el detector de caras de Viola-Jones
bbox = step(faceDetector, grayFrame);

% Definir un ROI para la boca basado en la detección de la cara
% Ajustar estos valores según la posición esperada de la boca en el cuadro detectado
%mouthROI = bbox;
%mouthROI(2) = bbox(2) + bbox(4)/1.9; % Mover el ROI hacia la parte inferior de la cara
%mouthROI(4) = bbox(4)/2.5;             % Reducir la altura del ROI

%Definir un ROI sin las orejas
mouthROI = bbox;
mouthROI(1)=bbox(1)+bbox(1)/16;
mouthROI(2) = bbox(2) + bbox(4)/1.9; % Mover el ROI hacia la parte inferior de la cara
mouthROI(3)=bbox(3)-(bbox(3)/3.6);
mouthROI(4) = bbox(4)/2.5;   

%Definir un ROI con las orejas
%mouthROI = bbox;
%mouthROI(1)=bbox(1)+bbox(1)/20;
%mouthROI(2) = bbox(2) + bbox(4)/1.9; % Mover el ROI hacia la parte inferior de la cara
%mouthROI(3)=bbox(3)-bbox(3)/4.1;
%mouthROI(4) = bbox(4)/2.5;   


% Mostrar la imagen y el ROI de la boca
figure;
imshow(videoFrame);
rectangle('Position', mouthROI, 'EdgeColor', 'r'); 
title('Detected face which ROI is in mouthROI')

% Detectar puntos críticos solo en la región de la boca

points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', mouthROI);


% Mostrar los puntos críticos detectados
hold on;
plot(points);

% Opcional: mejorar la imagen de la boca aplicando preprocesamiento
% Extraer la región de la boca
mouthRegion = imcrop(grayFrame, mouthROI);

% Aplicar aumento de contraste (histogram equalization)
mouthRegion = histeq(mouthRegion);

%
%mouthRegionAdjusted = imadjust(histeq(mouthRegion), [0.005 1], []);
%mouthRegion=mouthRegionAdjusted;

% Detectar puntos críticos mejorados en la región de la boca procesada
pointsEnhanced = detectMinEigenFeatures(mouthRegion);


% Mostrar los puntos críticos mejorados
figure;
imshow(mouthRegion);
hold on;
title('Imagen ecualizada con los puntos mejorados');
plot(pointsEnhanced); 


% Usar los puntos mejorados para el seguimiento
points = pointsEnhanced;
locations_points = points.Location;
valuetoADD= [mouthROI(1), mouthROI(2)];
new_locations_points = locations_points+ valuetoADD;
newPoints = cornerPoints(new_locations_points);
points=newPoints;

% Inicializa los puntos de la caja delimitadora
bboxPoints = bbox2points(bbox);

%%
% Initialize a tracker to track the points
pointTracker = vision.PointTracker('BlockSize',[63 63]); % 'MaxBidirectionalError', 15

% Initialize the tracker with the initial point locations and the initial
% video frame.
points = points.Location;
num_points_iniciales= size(points,1);
initialize(pointTracker, points, videoFrame);

%%
% Initialize Video player to display the results
videoPlayer = vision.VideoPlayer('Position', ...
    [100 100 [size(videoFrame, 2), size(videoFrame, 1)] + 30]);
%%
% Track the face
oldPoints = points;
frame_count=0
all_pointsCells=cell(numFrames,1);
numericVector=zeros (num_points_iniciales,1 )

while hasFrame(videoReader); 
    % get the next frame
    videoFrame = readFrame(videoReader); 
    frame_count=frame_count+1;
    

    % Track the points. Note that some points may be lost.
    [points, isFound] = step(pointTracker, videoFrame); %pongo un 1L en el vector isFound cuando cada uno de los puntos rastreados fue encontrado en el frame actual 
    numericVector= double(isFound);
    nuevoVector=zeros (num_points_iniciales, 1)

    for i=1:length(isFound)
        
        if isFound(i)
            nuevoVector(i)=1;
        else
            nuevoVector(i)=0;
        end
    end
    statusVectors{frame_count}=nuevoVector;

    visiblePoints = points(isFound, :); %visiblePoints guarda solo los puntos que fueron encontrados en el frame actual
    oldInliers = oldPoints(isFound, :);% oldInliers guarda los puntos del frame anterior que fueron encontrados en el actual

    if size(visiblePoints, 1) >= 2 % need at least 2 points

        % Estimate the geometric transformation between the old points
        % and the new points and eliminate outliers
        [xform, inlierIdx] = estgeotform2d( ...
            oldInliers, visiblePoints, 'similarity', 'MaxDistance', 1e5,'Confidence',90); %aumentar el valor lo vuelve menos restrictivo
        oldInliers = oldInliers(inlierIdx, :);
        visiblePoints = visiblePoints(inlierIdx, :);

        % Apply the transformation to the bounding box points
        bboxPoints = transformPointsForward(xform, bboxPoints);

        % Insert a bounding box around the object being tracked
        bboxPolygon = reshape(bboxPoints', 1, []);
        videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, ...
            'LineWidth', 2);

        % Display tracked points
        videoFrame = insertMarker(videoFrame, visiblePoints, '+', ...
            'Color', 'green');

        % Reset the points
        oldPoints = visiblePoints;
        setPoints(pointTracker, oldPoints);
    end

    % Display the annotated video frame using the video player object
    step(videoPlayer, videoFrame);
   

    startIndx=(frame_count-1)*size(visiblePoints,1)+1;
    endIndx= (frame_count*(size(visiblePoints,1)));
    all_points(startIndx:endIndx,:)=visiblePoints;

    all_pointsCells{frame_count} = visiblePoints; %almaceno cuantos puntos hay en cada frame

end

visiblePoints_150 = all_pointsCells{150};

% Clean up
release(videoPlayer);
release(pointTracker);
%%
%vector=[1; 1; 7; 1];
%nuevo_vector=zeros(6,1)
%nuevo_vector=zeros(size(vector,1),1)

%for i=1:size(vector,1)
    %if vector(i)==1
        %nuevo_vector(i,:)=vector(i)
    
    %else
        %nuevo_vector(i,:)=0
    %end
%end

