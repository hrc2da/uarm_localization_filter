%% pink-green
rows = 2160;
columns = 3840;
black = zeros(rows, columns, 'uint8');
redRamp = uint8(linspace(0, 255, columns));
blueRamp = uint8(linspace(0, 255, columns));
greenRamp = uint8(linspace(0, 255, rows))';
% Make into 2-D image.
redRamp = repmat(redRamp, [rows, 1]);
blueRamp = repmat(blueRamp, [rows, 1]);
greenRamp = repmat(greenRamp, [1, columns]);

rgbImage = cat(3, greenRamp, blueRamp, greenRamp);
imshow(rgbImage);
imwrite(rgbImage, 'grad1.jpg');

%% Blue-yellow
rows = 2160;
columns = 3840;

Ramp1 = uint8(linspace(0, 255, columns));
Ramp2 = uint8(linspace(0, 255, columns));
Ramp3 = uint8(linspace(0, 255, rows))';
% Make into 2-D image.
Ramp1 = repmat(Ramp1, [rows, 1]);
Ramp2 = repmat(Ramp2, [rows, 1]);
Ramp3 = repmat(Ramp3, [1, columns]);

rgbImage = cat(3, Ramp1, Ramp2, Ramp3);
imshow(rgbImage);
imwrite(rgbImage, 'grad2.jpg');

%% Blue-red
rows = 2160;
columns = 3840;
black = zeros(rows, columns, 'uint8');
Ramp1 = uint8(linspace(0, 255, columns));
Ramp2 = uint8(linspace(0, 255, columns));
Ramp3 = uint8(linspace(0, 255, rows))';
% Make into 2-D image.
Ramp1 = repmat(Ramp1, [rows, 1]);
Ramp2 = repmat(Ramp2, [rows, 1]);
Ramp3 = repmat(Ramp3, [1, columns]);

rgbImage = cat(3, Ramp3, Ramp1, Ramp1);
imshow(rgbImage);
imwrite(rgbImage, 'grad3.jpg');

%% Rainbow
%https://colorswall.com/palette/102/
rows = 450*7;
columns = rows*2;

r = uint8([255*ones(1,900*3) zeros(1,900*2) 75*ones(1,900) 238*ones(1,900)]);
g = uint8([zeros(1, 900) 130*ones(1,900) 255*ones(1,900) 128*ones(1,900) zeros(1,2*900) 130*ones(1,900)]);
b = uint8([zeros(1, 4*900) 255*ones(1,900) 130*ones(1,900) 238*ones(1,900)]);

red = zeros(rows, columns);
green = zeros(rows, columns);
blue = zeros(rows, columns);

for i=1:rows
    red(i,:) = r;%*(i/rows);
    green(i,:) = g;%*(i/rows);
    blue(i,:) = b;%*(i/rows);
end

rgbImage = cat(3, red, green, blue);
imshow(rgbImage);
imwrite(rgbImage, 'grad4.jpg');
% well that didn't work

%% crazy

rows = 2160;
columns = 3840;
black = zeros(rows, columns, 'uint8');
Ramp1 = uint8([linspace(0, 200, columns/4) linspace(200, 0, columns/4) linspace(0, 255, columns/4) linspace(255, 0, columns/4)]);
Ramp2 = uint8(linspace(0, 255, columns));
Ramp3 = uint8(linspace(0, 255, rows))';
% Make into 2-D image.
Ramp1 = repmat(Ramp1, [rows, 1]);
Ramp2 = repmat(Ramp2, [rows, 1]);
Ramp3 = repmat(Ramp3, [1, columns]);

rgbImage = cat(3, Ramp1, Ramp2, Ramp3);
imshow(rgbImage);
imwrite(rgbImage, 'grad5.jpg');

%% crazy 2

rows = 2160;
columns = 3840;
black = zeros(rows, columns, 'uint8');
Ramp1 = uint8([linspace(0, 200, columns/4) linspace(200, 0, columns/4) linspace(0, 255, columns/4) linspace(255, 0, columns/4)]);
Ramp2 = uint8(linspace(0, 255, columns));
Ramp3 = uint8([linspace(0, 255, rows/2) linspace(255, 0, rows/2)])';
% Make into 2-D image.
Ramp1 = repmat(Ramp1, [rows, 1]);
Ramp2 = repmat(Ramp2, [rows, 1]);
Ramp3 = repmat(Ramp3, [1, columns]);

rgbImage = cat(3, Ramp1, Ramp2, Ramp3);
imshow(rgbImage);
imwrite(rgbImage, 'grad6.jpg');

%% crazy 3

rows = 2160;
columns = 3840;
black = zeros(rows, columns, 'uint8');
Ramp1 = uint8([linspace(0, 255, columns/6) linspace(255, 0, columns/6) linspace(0, 255, columns/6) linspace(255, 0, columns/6) linspace(0, 255, columns/6) linspace(255, 0, columns/6)]);
Ramp2 = uint8(linspace(0, 255, columns));
Ramp3 = uint8(linspace(0, 255, rows))';
% Make into 2-D image.
Ramp1 = repmat(Ramp1, [rows, 1]);
Ramp2 = repmat(Ramp2, [rows, 1]);
Ramp3 = repmat(Ramp3, [1, columns]);

rgbImage = cat(3, Ramp1, Ramp2, Ramp3);
imshow(rgbImage);
imwrite(rgbImage, 'grad7.jpg');

%% crazy 4

rows = 2160;
columns = 3840;
black = zeros(rows, columns, 'uint8');
Ramp1 = uint8([linspace(0, 255, columns/6) linspace(255, 0, columns/6) linspace(0, 255, columns/6) linspace(255, 0, columns/6) linspace(0, 255, columns/6) linspace(255, 0, columns/6)]);
Ramp2 = uint8([linspace(0, 200, columns/4) linspace(200, 0, columns/4) linspace(0, 255, columns/4) linspace(255, 0, columns/4)]);
Ramp3 = uint8([linspace(0, 255, rows/2) linspace(255, 0, rows/2)])';
% Make into 2-D image.
Ramp1 = repmat(Ramp1, [rows, 1]);
Ramp2 = repmat(Ramp2, [rows, 1]);
Ramp3 = repmat(Ramp3, [1, columns]);

rgbImage = cat(3, Ramp1, Ramp2, Ramp3);
imshow(rgbImage);
imwrite(rgbImage, 'grad7.jpg');

%% blocky

rows = 6000;
columns = rows*2;
Ramp1 = [];
Ramp2 = [];

for i=0:28:255
    Ramp1 = [Ramp1 i*ones(1, columns/10)];
end

for i=0:50:255
    Ramp2 = [Ramp2 i*ones(1, rows/6)];
end

Ramp1 = uint8(Ramp1);
Ramp2 = uint8(Ramp2)';

Ramp1 = repmat(Ramp1, [rows, 1]);
Ramp2 = repmat(Ramp2, [1, columns]);

rgbImage = cat(3, Ramp1, Ramp2, Ramp2);
imshow(rgbImage);
imwrite(rgbImage, 'grad8.jpg');
