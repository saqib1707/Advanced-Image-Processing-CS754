addpath('lib');
clear
close all
clc

BUFFER_SIZE = 807;
l_size = 100;
%L = zeros(1,l_size);


Sample_size = zeros(4, 2*BUFFER_SIZE);
training_window_size = zeros(4, 2*BUFFER_SIZE);
Adaptive_Threshold = zeros(4,2*BUFFER_SIZE);
Accel_data = zeros(4,BUFFER_SIZE);
Accel_motion_data = zeros(4, BUFFER_SIZE);
Accel_motion_data_cpy = zeros(4, BUFFER_SIZE);
differentiation_1 = zeros(4, BUFFER_SIZE);
integeral_1 = zeros(3,BUFFER_SIZE);
mean = zeros(3,1);
diff = zeros(3,1);
myvar=zeros(3,1);
mystd=zeros(3,1);
sum_x = 0;
sum_y = 0;
sum_z = 0;

alpha = 0.02;
k = 3; 
size = 0;
j  = 0;

% get accel x,y,z axis data from stored file

initial_training_window_size = 100;
x_itr = 1;
y_itr = 1;

	training_window_size(1,x_itr) = initial_training_window_size;
	x_itr += 1;
	training_window_size(2,y_itr) = initial_training_window_size;
	y_itr += 1;

	
Accel_data(1,:) = (((dlmread("data3.txt")(:,1))'))(1:BUFFER_SIZE);
Accel_data(2,:) = (((dlmread("data3.txt")(:,2))'))(1:BUFFER_SIZE);
Accel_data(3,:) = (((dlmread("data3.txt")(:,3))'))(1:BUFFER_SIZE);
Accel_data(4,:) = ((dlmread("data3.txt")(:,4))')(1:BUFFER_SIZE);


%=================================================================%

MEAN = zeros(3,1);
STD = zeros(3,1);
VAR = zeros(3,1);
SUM  = zeros(3,1);

for i = 1:initial_training_window_size
	SUM(1,1) += abs(Accel_data(1,i));
	SUM(2,1) += abs(Accel_data(2,i));
end

MEAN(1,1) = SUM(1,1)/initial_training_window_size;
MEAN(2,1) = SUM(2,1)/initial_training_window_size;

DIFF = zeros(3,1);
for i = 1:initial_training_window_size
	DIFF(1,1) = Accel_data(1,i) - MEAN(1,1);
	DIFF(2,1) = Accel_data(2,i) - MEAN(2,1);
	
	VAR(1,1) += (DIFF(1,1) * DIFF(1,1));
	VAR(2,1) += (DIFF(2,1) * DIFF(2,1));
end

STD(1,1) = sqrt(VAR(1,1)/initial_training_window_size);
STD(2,1) = sqrt(VAR(2,1)/initial_training_window_size);

thresh_accel_x = MEAN(1,1) + k*STD(1,1);
thresh_accel_y = MEAN(2,1) + k*STD(2,1);

disp("SUM_X"); disp(SUM(1,1));
disp("MEAN_X"); disp(MEAN(1,1));
disp("STD_X"); disp(STD(1,1));
disp("Accel_data threshold_X"); disp(thresh_accel_x);

disp("SUM_Y"); disp(SUM(2,1));
disp("MEAN_Y"); disp(MEAN(2,1));
disp("STD_Y"); disp(STD(2,1));
disp("Accel_data threshold_Y"); disp(thresh_accel_y);
%=================================================================%


%Accel_motion_data(1:4,:) = Accel_data(1:4,:);

for i = 2:initial_training_window_size
	training_window_size(1,x_itr) = initial_training_window_size;
	x_itr += 1;
	training_window_size(2,y_itr) = initial_training_window_size;
	y_itr += 1;
	
	differentiation_1(1,i) = (Accel_data(1,i) - Accel_data(1,i-1));
	differentiation_1(2,i) = (Accel_data(2,i) - Accel_data(2,i-1));
	
	sum_x = sum_x + abs(differentiation_1(1,i));
	sum_y = sum_y + abs(differentiation_1(2,i));
end

mean(1,1) = sum_x/(initial_training_window_size-1);
mean(2,1) = sum_y/(initial_training_window_size-1);
disp("sum_x = "), disp(sum_x);
disp("mean_x = "), disp(mean(1,1));

for i = 1:initial_training_window_size
	diff(1,1)=differentiation_1(1,i)-mean(1,1);
	diff(2,1)=differentiation_1(2,i)-mean(2,1);
	
	myvar(1,1)+=diff(1,1)^2;
	myvar(2,1)+=diff(2,1)^2;
end

mystd(1,1)=sqrt((myvar(1,1)/initial_training_window_size));
disp("mystd = "), disp(mystd(1,1));
mystd(2,1)=sqrt((myvar(2,1)/initial_training_window_size));

threshold_x = (k * mystd(1,1) + mean(1,1));
threshold_y = (k * mystd(2,1) + mean(2,1));
disp("difference data threshold_x"); disp(threshold_x);
disp("difference data threshold_y"); disp(threshold_y);

temp_th = threshold_x;
temp_mean = mean(1,1);





next_training_window_size = zeros(3,1);
next_training_window_size_temp = zeros(3,1);

next_training_window_size(1,1) = log(2/alpha)* (1/(2 * (mean(1,1) + k*mystd(1,1))^2));
next_training_window_size(1,1) = ceil(next_training_window_size(1,1));
disp("next_training_window_size_x = "), disp(next_training_window_size(1,1));
training_window_size(1,x_itr) = next_training_window_size(1,1);
x_itr += 1;

next_training_window_size(2,1) = log(2/alpha)* (1/(2 * (mean(2,1) + k*mystd(2,1))^2));
next_training_window_size(2,1) = ceil(next_training_window_size(2,1));
disp("next_training_window_size_y = "), disp(next_training_window_size(2,1));
training_window_size(2,y_itr) = next_training_window_size(2,1);
y_itr += 1;


next_training_window_size_temp(1,1) = next_training_window_size(1,1);
next_training_window_size_temp(2,1) = next_training_window_size(2,1);



%====================================================================X-AXIS===========================================================%
j = training_window_size(1,1) + 1;
while(j <= BUFFER_SIZE)
	
	temp_mean = mean(1,1);
	temp_th = threshold_x;
	
	sum_x = 0;
	mean(1,1) = 0;
	myvar(1,1) = 0;
	mystd(1,1) = 0;
	size = 0;
	
	start = j;
		%disp("j = "), disp(j);
	while(j <= BUFFER_SIZE && size < next_training_window_size(1,1))
		training_window_size(1,x_itr) = next_training_window_size_temp(1,1);
		Adaptive_Threshold(1,x_itr) = temp_th;
		x_itr += 1;
		differentiation_1(1,j) = (Accel_data(1,j) - Accel_data(1,j-1));
		if(abs(differentiation_1(1,j)) < threshold_x)
			sum_x = sum_x + abs(differentiation_1(1,j));
			size += 1;
			%Accel_motion_data(1,j) = 0;
		end
		%{
		if(abs(differentiation_1(1,j)) < threshold_x && abs(Accel_data(1,j)) < thresh_accel_x)
			Accel_motion_data(1,j) = 0;
		end
		if(abs(differentiation_1(1,j)) < threshold_x && abs(Accel_data(1,j)) > thresh_accel_x)
			Accel_motion_data(1,j) = Accel_motion_data(1,j-1);
		end
		if(abs(differentiation_1(1,j)) > threshold_x && abs(Accel_data(1,j)) > thresh_accel_x)
			Accel_motion_data(1,j) = Accel_data(1,j) - MEAN(1,1);
		end
		%}
		if(abs(differentiation_1(1,j)) < threshold_x)
			Accel_motion_data(1,j) = 0;
		end
		if(abs(differentiation_1(1,j)) > threshold_x)
			Accel_motion_data(1,j) = Accel_data(1,j) - MEAN(1,1);
		end
		
		j = j + 1;
	endwhile
	
	%disp("sum = "), disp(sum_x);
	mean(1,1) = sum_x/(size);
	%disp("mean = "), disp(mean(1,1));

	size = 0;
	j = start;
		%disp("j new = "), disp(j);
	while(j <= BUFFER_SIZE && size < next_training_window_size(1,1))
		if(abs(differentiation_1(1,j)) < threshold_x)	
			diff(1,1) = differentiation_1(1,j)-mean(1,1);
			myvar(1,1) += diff(1,1)^2;
			size += 1;
		end
		j = j + 1;
	endwhile
	%disp("size = "), disp(size);
	
	mystd(1,1)=sqrt((myvar(1,1)/(size)));
	%disp("mystd = "), disp(mystd(1,1));

	threshold_x = (k * mystd(1,1) + mean(1,1));

	next_training_window_size_temp(1,1) = next_training_window_size(1,1);
	next_training_window_size(1,1) = log(2/alpha) * (1/(2 * (mean(1,1) + k*mystd(1,1))^2));
	next_training_window_size(1,1) = ceil(next_training_window_size(1,1));
	%disp("next_training_window_size_x = "), disp(next_training_window_size(1,1));
	
	if(threshold_x == 0 || next_training_window_size(1,1) <= 2)
		disp("threshold is zero or next_training_window_size_x is zero");
		next_training_window_size(1,1) = 500;
		threshold_x = temp_th;
		mean(1,1) = temp_mean;
		continue;
	end;
	
	next_training_window_size_temp(1,1) = next_training_window_size(1,1);
endwhile
%====================================================================X-AXIS===========================================================%



%====================================================================Y-AXIS===========================================================%

temp_th = threshold_y;
temp_mean = mean(2,1);

j = training_window_size(2,1) + 1;
while(j <= BUFFER_SIZE)
	
	temp_mean = mean(2,1);
	temp_th = threshold_y;
	
	sum_y = 0;
	mean(2,1) = 0;
	myvar(2,1) = 0;
	mystd(2,1) = 0;
	size = 0;
	
	start = j;
		%disp("j = "), disp(j);
	while(j <= BUFFER_SIZE && size < next_training_window_size(2,1))
		training_window_size(2,y_itr) = next_training_window_size_temp(2,1);
		Adaptive_Threshold(2,y_itr) = temp_th;
		y_itr += 1;
		
		differentiation_1(2,j) = (Accel_data(2,j) - Accel_data(2,j-1));
		if(abs(differentiation_1(2,j)) < threshold_y)
			sum_y = sum_y + abs(differentiation_1(2,j));
			size += 1;
			%Accel_motion_data(1,j) = 0;
		end
		%{
		if(abs(differentiation_1(2,j)) < threshold_y && abs(Accel_data(2,j)) < thresh_accel_y)
			Accel_motion_data(2,j) = 0;
		end
		if(abs(differentiation_1(2,j)) < threshold_y && abs(Accel_data(2,j)) > thresh_accel_y)
			Accel_motion_data(2,j) = Accel_motion_data(2,j-1);
		end
		if(abs(differentiation_1(2,j)) > threshold_y && abs(Accel_data(2,j)) > thresh_accel_y)
			Accel_motion_data(2,j) = Accel_data(2,j) - MEAN(2,1);
		end
		%}
		if(abs(differentiation_1(2,j)) < threshold_y)
			Accel_motion_data(2,j) = 0;
		end
		if(abs(differentiation_1(2,j)) > threshold_y)
			Accel_motion_data(2,j) = Accel_data(2,j) - MEAN(2,1);
		end
		
		j = j + 1;
	endwhile
	
	%disp("sum = "), disp(sum_x);
	mean(2,1) = sum_y/(size);
	%disp("mean = "), disp(mean(1,1));

	size = 0;
	j = start;
		%disp("j new = "), disp(j);
	while(j <= BUFFER_SIZE && size < next_training_window_size(2,1))
		if(abs(differentiation_1(2,j)) < threshold_y)	
			diff(2,1) = differentiation_1(2,j)-mean(2,1);
			myvar(2,1) += diff(2,1)^2;
			size += 1;
		end
		j = j + 1;
	endwhile
	%disp("size = "), disp(size);
	
	mystd(2,1)=sqrt((myvar(2,1)/(size)));
	%disp("mystd = "), disp(mystd(1,1));

	threshold_y = (k * mystd(2,1) + mean(2,1));

	next_training_window_size_temp(2,1) = next_training_window_size(2,1);
	next_training_window_size(2,1) = log(2/alpha) * (1/(2 * (mean(2,1) + k*mystd(2,1))^2));
	next_training_window_size(2,1) = ceil(next_training_window_size(2,1));
	%disp("next_training_window_size_x = "), disp(next_training_window_size(1,1));
	
	if(threshold_y == 0 || next_training_window_size(2,1) <= 2)
		disp("threshold is zero or next_training_window_size_y is zero");
		next_training_window_size(2,1) = 500;
		threshold_y = temp_th;
		mean(2,1) = temp_mean;
		continue;
	end;
	
	next_training_window_size_temp(2,1) = next_training_window_size(2,1);
endwhile
%====================================================================Y-AXIS===========================================================%


%Hoeffding_convergence
hfig=(figure);
scrsz = get(0,'ScreenSize');
set(hfig,'position',scrsz);


%{
subplot(4,1,1)
p1 = plot(1:l_size, L(1,:),'k');
hold on;
grid on;
ylabel ("L - sample-size");
title(['Sajal']);
%}

subplot(4,1,1)
p1 = plot(1:BUFFER_SIZE, Accel_data(1,:),'r');
hold on;
grid on;
p2 = plot(1:BUFFER_SIZE, Accel_data(2,:),'b');
hold on;
grid on;
ylabel ("Accel Input");
title(['Accel Input with motion and non-motion areas']);

subplot(4,1,2)
p1 = plot(1:BUFFER_SIZE, training_window_size(1,1:BUFFER_SIZE),'r');
hold on;
grid on;
p2 = plot(1:BUFFER_SIZE, training_window_size(2,1:BUFFER_SIZE),'b');
hold on;
grid on;
ylabel ("Training Window Size");
title(['Training Window Size']);

subplot(4,1,3)
p1 = plot(1:BUFFER_SIZE, Adaptive_Threshold(1,1:BUFFER_SIZE),'r');
hold on;
grid on;
p2 = plot(1:BUFFER_SIZE, Adaptive_Threshold(2,1:BUFFER_SIZE),'b');
hold on;
grid on;
ylabel ("Adaptive Threshold");
title(['Adaptive Threshold']);

Accel_motion_data(2,:)
subplot(4,1,4)
p1 = plot(1:BUFFER_SIZE, Accel_motion_data(1,:),'r');
hold on;
grid on;
p2 = plot(1:BUFFER_SIZE, Accel_motion_data(2,:),'b');
hold on;
grid on;
ylabel ("Motion Identified Accel Output");
title(['Motion Identified Accel Output']);

hfig=(figure);
scrsz = get(0,'ScreenSize');
set(hfig,'position',scrsz);

subplot(2,1,1)
p1 = plot(1:BUFFER_SIZE, Accel_data(1,1:BUFFER_SIZE),'r');
hold on;
grid on;
p2 = plot(1:BUFFER_SIZE, Accel_data(2,1:BUFFER_SIZE),'b');
hold on;
grid on;
ylabel ("Zoomed in Accel-data");
title(['Zoomed in Accel-data']);

subplot(2,1,2)
p1 = plot(1:BUFFER_SIZE, Accel_motion_data(1,1:BUFFER_SIZE),'r');
hold on;
grid on;
p2 = plot(1:BUFFER_SIZE, Accel_motion_data(2,1:BUFFER_SIZE),'b');
hold on;
grid on;
ylabel ("Zoomed in Accel-motion-data");
title(['Zoomed in Accel-motion-data']);
