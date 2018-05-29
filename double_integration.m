clear all
close all
clc
data = readcsv('STennis_b.csv');
%time = data(:,7);
acc = data(:,1);
time = 1:size(acc,1);
time = time';
%time = load('D:\Users\Desktop\time.txt');
%acc = load('D:\Users\Desktop\data.txt');
figure
plot(time,acc)
xlabel('Time (sec)')
ylabel('Acceleration (mm/sec^2)')
%% Design Low Pass Filter
fs = 100; % Sampling Rate
fc = 0.5; % Cut off Frequency
order = 9; % 6th Order Filter
%% Filter Acceleration Signals
[b1 a1] = butter(order,fc);
accf=filtfilt(b1,a1,acc);
figure(2)
plot(time,accf,'r'); %hold on
%plot(time,acc)
xlabel('Time (sec)')
ylabel('Filtered Acceleration (mm/sec^2)')
%% First Integration (Acceleration - Veloicty)
velocity=cumtrapz(time,accf);
figure(3)
plot(time,velocity)
xlabel('Time (sec)')
ylabel('Velocity (mm/sec)')
%% Filter Velocity Signals
[b2 a2] = butter(order,fc);
velf = filtfilt(b2,a2,velocity);
%% Second Integration (Velocity - Displacement)
Displacement=cumtrapz(time, velf);
figure(4)
plot(time,Displacement)
xlabel('Time (sec)')
ylabel('Displacement (mm)')