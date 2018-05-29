addpath('lib');
clear
close all
clc

alpha = 0.8;

GRAVITY = 9.80665;
RAD2DEG = 57.2957795;
DEG2RAD = 1 / RAD2DEG;
PI = 3.141593;
%EETA = 0.05 * PI;
EETA = 0.1;
NON_ZERO_VAL = 0.025;   %%%%%%%%%%%%%% Check
ROUNDOFF_VAL = 5;	%%%%%%%%%%%%%% Check
PI_DEG = 180;
US2S =  1.0 / 1000000.0;

Max_Range_Accel = 39.203407; Min_Range_Accel = -39.204006; Res_Accel = 0.000598;
Max_Range_Gyro = 1146.862549; Min_Range_Gyro = -1146.880005; Res_Gyro = 0.017500;

BUFFER_SIZE = 200;
TRAINING_SIZE = 100;

acc_e = [0.0;0.0;1.0]; % gravity vector in earth frame

Accel_data = zeros(4,BUFFER_SIZE);
Accel_data_in = zeros(4,BUFFER_SIZE);
decision_making = zeros(4,BUFFER_SIZE);
sajal_Accel_data = zeros(BUFFER_SIZE,3);
Gravity_data = zeros(3,BUFFER_SIZE);
differentiation_1 = zeros(3,BUFFER_SIZE);
differentiation_1_2 = zeros(3,BUFFER_SIZE);
integeral_1 = zeros(3,BUFFER_SIZE);
integeral_1_2 = zeros(3,BUFFER_SIZE);
integeral_2 = zeros(3,BUFFER_SIZE);
integeral_2_2 = zeros(3,BUFFER_SIZE);
integeral_3 = zeros(3,BUFFER_SIZE);
integeral_3_2 = zeros(3,BUFFER_SIZE);

Accel_data_in(1,:) = ((dlmread("o_fast_template.txt")(:,1))')(1:BUFFER_SIZE);
Accel_data_in(2,:) = ((dlmread("o_fast_template.txt")(:,2))')(1:BUFFER_SIZE);
Accel_data_in(3,:) = ((dlmread("o_fast_template.txt")(:,3))')(1:BUFFER_SIZE);
#Accel_data_in(4,:) = ((dlmread("o_10ms_summed.txt")(:,4))')(1:BUFFER_SIZE);


Accel_data = Accel_data_in(1:3,:);


%Accel_data_in_summed = zeros(3,BUFFER_SIZE/5);
%
%for i = 1 : (BUFFER_SIZE-5)/5
%  for j = 1 : 5
%    Accel_data_in_summed(1,i) =  Accel_data_in_summed(1,i) + Accel_data_in(1,i*5 + j);
%    Accel_data_in_summed(2,i) =  Accel_data_in_summed(1,i) + Accel_data_in(2,i*5 + j);
%    Accel_data_in_summed(3,i) =  Accel_data_in_summed(1,i) + Accel_data_in(3,i*5 + j);
%  end
%end
%
%BUFFER_SIZE  = BUFFER_SIZE/5;
%
%
%Accel_data = Accel_data_in_summed(1:3,:);

sum_x = 0;
sum_y = 0;
sum_z = 0;

for i = 2:TRAINING_SIZE

differentiation_1(1,i) = (Accel_data(1,i) - Accel_data(1,i-1));
differentiation_1(2,i) = (Accel_data(2,i) - Accel_data(2,i-1));
differentiation_1(3,i) = (Accel_data(3,i) - Accel_data(3,i-1));

%{
disp("differentiation_1x = "), disp(differentiation_1(1,i));
disp("differentiation_1y = "), disp(differentiation_1(2,i));
disp("differentiation_1z = "), disp(differentiation_1(3,i));
%}

sum_x = sum_x + abs(differentiation_1(1,i));
sum_y = sum_y + abs(differentiation_1(2,i));
sum_z = sum_z + abs(differentiation_1(3,i));

end

%======================================z-score method==============================%
MEAN_X = 0.0;
MEAN_Y = 0.0;
MEAN_Z = 0.0;

SUM_X = 0.0;
SUM_Y = 0.0;
SUM_Z = 0.0;

for i = 2:BUFFER_SIZE
	SUM_X = SUM_X + Accel_data(1,i);
	SUM_Y = SUM_Y + Accel_data(2,i);
	SUM_Z = SUM_Z + Accel_data(3,i);
end

	MEAN_X = SUM_X/(BUFFER_SIZE-1);
	MEAN_Y = SUM_Y/(BUFFER_SIZE-1);
	MEAN_Z = SUM_Z/(BUFFER_SIZE-1);
%======================================z-score method==============================%

mean = zeros(3,1);
diff = zeros(3,1);

mean(1,1) = sum_x/(TRAINING_SIZE-1);
mean(2,1) = sum_y/(TRAINING_SIZE-1);
mean(3,1) = sum_z/(TRAINING_SIZE-1);

myvar=zeros(3,TRAINING_SIZE);
mystd=zeros(3,TRAINING_SIZE);

for i = 1:TRAINING_SIZE
	
	diff(1,1)=differentiation_1(1,i)-mean(1,1);
	diff(2,1)=differentiation_1(2,i)-mean(2,1);
	diff(3,1)=differentiation_1(3,i)-mean(3,1);
	
    myvar(1,1)+=diff(1,1)^2;
	myvar(2,1)+=diff(2,1)^2;
	myvar(3,1)+=diff(3,1)^2;
	
end

    mystd(1,1)=sqrt((myvar(1,1)/(TRAINING_SIZE-1)));
    mystd(2,1)=sqrt((myvar(2,1)/(TRAINING_SIZE-1)));
    mystd(3,1)=sqrt((myvar(3,1)/(TRAINING_SIZE-1)));

threshold_x = (3 * mystd(1,1) + mean(1,1))
threshold_y = (3 * mystd(2,1) + mean(2,1))
threshold_z = (3 * mystd(3,1) + mean(3,1))

disp("sum_x = "), disp(sum_x);
disp("sum_y = "), disp(sum_y);
disp("sum_z = "), disp(sum_z);

disp("mean_x = "), disp(mean(1,1));
disp("mean_y = "), disp(mean(2,1));
disp("mean_z = "), disp(mean(3,1));

disp("mystd_x = "), disp(mystd(1,1));
disp("mystd_y = "), disp(mystd(2,1));
disp("mystd_z = "), disp(mystd(3,1));

	
for i = 2:BUFFER_SIZE
	differentiation_1(1,i) = (Accel_data(1,i) - Accel_data(1,i-1));
	differentiation_1(2,i) = (Accel_data(2,i) - Accel_data(2,i-1));
	differentiation_1(3,i) = (Accel_data(3,i) - Accel_data(3,i-1));
	
	if(abs(differentiation_1(1,i)) < threshold_x)
		differentiation_1(1,i) = 0.0;
	end
	
	if(abs(differentiation_1(2,i)) < threshold_y)
		differentiation_1(2,i) = 0.0;
	end
	
	if(abs(differentiation_1(3,i)) < threshold_z)
		differentiation_1(3,i) = 0.0;
	end
	
	integeral_1(1,i) = integeral_1(1,i-1) + differentiation_1(1,i);
	integeral_1(2,i) = integeral_1(2,i-1) + differentiation_1(2,i);
	integeral_1(3,i) = integeral_1(3,i-1) + differentiation_1(3,i);
	
	if(abs(differentiation_1(1,i)) < threshold_x)
		integeral_1(1,i) = 0.0;
	end
	
	
	if(abs(differentiation_1(2,i)) < threshold_y)
		integeral_1(2,i) = 0.0;
	end

	
	if(abs(differentiation_1(3,i)) < threshold_z)
		integeral_1(3,i) = 0.0;
	end
	
	if (i <= 3)
		% to stabilize the values
		differentiation_1(1,i) = 0.0;
		integeral_1(1,i) = 0.0;
		integeral_2(1,i) = 0.0;
		differentiation_1(2,i) = 0.0;
		integeral_1(2,i) = 0.0;
		integeral_2(2,i) = 0.0;
		differentiation_1(3,i) = 0.0;
		integeral_1(3,i) = 0.0;
		integeral_2(3,i) = 0.0;
	end
end

temp = zeros(4,BUFFER_SIZE);
temp(1,:) = Accel_data(1,:);
temp(2,:) = Accel_data(2,:);
temp(3,:) = Accel_data(3,:);

for i = 2 : BUFFER_SIZE

	if(integeral_1(1,i) == 0.0)
		Accel_data(1,i) = 0.0;
	else
		Accel_data(1,i) = Accel_data(1,i) - MEAN_X;
	endif
	
	if(integeral_1(2,i) == 0.0)
		Accel_data(2,i) = 0.0;
	else
		Accel_data(2,i) = Accel_data(2,i) - MEAN_Y;
	endif
	
	if(integeral_1(3,i) == 0.0)
		Accel_data(3,i) = 0.0;
	else
		Accel_data(3,i) = Accel_data(3,i) - MEAN_Z;
	endif
end

for i = 2 : BUFFER_SIZE
	
	if(abs(differentiation_1(1,i)) < threshold_x)
		Accel_data(1,i) = 0.0;
	end
	
	if(abs(differentiation_1(2,i)) < threshold_y)
		Accel_data(2,i) = 0.0;
	end
	
	if(abs(differentiation_1(3,i)) < threshold_z)
		Accel_data(3,i) = 0.0;
	end
	
	integeral_2(1,i) = integeral_2(1,i-1)+ Accel_data(1,i);
	integeral_2(2,i) = integeral_2(2,i-1)+ Accel_data(2,i);
	integeral_2(3,i) = integeral_2(3,i-1)+ Accel_data(3,i);
	
	integeral_3(1,i) = integeral_3(1,i-1)+ integeral_2(1,i);
	integeral_3(2,i) = integeral_3(2,i-1)+ integeral_2(2,i);
	integeral_3(3,i) = integeral_3(3,i-1)+ integeral_2(3,i);
	
	if(abs(differentiation_1(1,i)) < threshold_x)
		integeral_2(1,i) = 0.0;
	end
	
	if(abs(differentiation_1(2,i)) < threshold_y)
		integeral_2(2,i) = 0.0;
	end
	
	if(abs(differentiation_1(3,i)) < threshold_z)
		integeral_2(3,i) = 0.0;
	end
	
	if (i <= 3)
		% to stabilize the values
		differentiation_1(1,i) = 0.0;
		integeral_1(1,i) = 0.0;
		integeral_2(1,i) = 0.0;
		differentiation_1(2,i) = 0.0;
		integeral_1(2,i) = 0.0;
		integeral_2(2,i) = 0.0;
		differentiation_1(3,i) = 0.0;
		integeral_1(3,i) = 0.0;
		integeral_2(3,i) = 0.0;
	end
end


% Accelerometer/Gyroscope scaled Plot results
hfig=(figure);
scrsz = get(0,'ScreenSize');
set(hfig,'position',scrsz);

UE = Accel_data(1:3,:);
UA = differentiation_1(1:3,:);
UB = integeral_1(1:3,:);
%UB = decision_making(1:3,:);
UC = integeral_2(1:3,:);
UD = integeral_3(1:3,:);

UH = temp(1:3,:);
%UP = UH(1,400:600)
%disp(UP)

subplot(5,1,1)
%p1 = plot(1:201,UP(:,:),'r');
p1 = plot(1:BUFFER_SIZE,UH(1,:),'r');
hold on;
grid on;
p2 = plot(1:BUFFER_SIZE,UH(2,:),'b');
hold on;
grid on;
ylabel ("Accelerometer Input");
legend([p1 p2],'X','Y');

subplot(5,1,2)
p1 = plot(1:BUFFER_SIZE,UE(1,:),'r');
hold on;
grid on;
p2 = plot(1:BUFFER_SIZE,UE(2,:),'b');
hold on;
grid on;
ylabel(['Motion Detected Values']);
legend([p1 p2],'X','Y');

subplot(5,1,3)
p1 = plot(1:BUFFER_SIZE,UB(1,:),'r');
hold on;
grid on;
p2 = plot(1:BUFFER_SIZE,UB(2,:),'b');
hold on;
grid on;
ylabel(['Bias Removed']);
legend([p1 p2],'X','Y');


subplot(5,1,4)
p1 = plot(1:BUFFER_SIZE,UC(1,:),'r');
hold on;
grid on;
p2 = plot(1:BUFFER_SIZE,UC(2,:),'b');
hold on;
grid on;
ylabel(['First Integeral']);
legend([p1 p2],'X','Y');

subplot(5,1,5)
p1 = plot(1:BUFFER_SIZE,UD(1,:),'r');
hold on;
grid on;
p2 = plot(1:BUFFER_SIZE,UD(2,:),'b');
hold on;
grid on;
ylabel(['Second Integeral']);
legend([p1 p2],'X','Y');

%=========================================================================%

hfig=(figure);
scrsz = get(0,'ScreenSize');
set(hfig,'position',scrsz);

subplot(1,1,1)
p1 = plot(UD(1,:),UD(2,:),'k');
hold on;
ylabel("Proposed Solution ");
title(['Alphabet-[b]']);
%=========================================================================%

%%=========================================================================%
%
%hfig=(figure);
%scrsz = get(0,'ScreenSize');
%set(hfig,'position',scrsz);
%
%subplot(1,1,1)
%p1 = plot(UD(2,:),UD(3,:),'k');
%hold on;
%ylabel("Proposed Solution2 ");
%title(['Alphabet-[b]']);
%%=========================================================================%
%
%
%%=========================================================================%
%
%hfig=(figure);
%scrsz = get(0,'ScreenSize');
%set(hfig,'position',scrsz);
%
%subplot(1,1,1)
%p1 = plot(UD(1,:),UD(3,:),'k');
%hold on;
%ylabel("Proposed Solution3 ");
%title(['Alphabet-[b]']);
%%=========================================================================%


