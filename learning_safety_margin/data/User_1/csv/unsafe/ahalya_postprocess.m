clear; clc;

% Import Panda robot wrapper
filepath = fileparts(which('ahalya_postprocess.m'));
addpath(genpath(fullfile(filepath, "robot-wrapper/")));
pandaRobot = PandaWrapper();

% Import data recorded on the real robot, and convert them to a format
% suitable for the practical.

datasetPath = fullfile(filepath, '5_19_22/ahalya_experiment/batch5-velocityLimits/');
disp(datasetPath)
filelist = dir(fullfile(datasetPath, '**', '*.db3'));
nTraj = length(filelist);

csvPath = fullfile(filepath, "5_19_22/csv_files_postprocessing/velocity_limits/fast/");
figure(1)
title('Fast Demonstrations')
%%
nSamplePoint = 200;
trajectories = nan(6, nSamplePoint, nTraj);
for iTraj=11:nTraj
    % Open new bag for new trajectory
    bagDirectory = filelist(iTraj).folder;
    bag = ros2bag(bagDirectory);
    msgs = readMessages(bag);

    msgsStruct = cell2mat(msgs);

    jointPosition = reshape(cell2mat({msgsStruct.position}), 7, []);
    position = pandaRobot.computeForwardKinematics(jointPosition);
    
    jointVelocity = reshape(cell2mat({msgsStruct.velocity}), 7, []);
    velocity = pandaRobot.computeForwardVelocity(jointPosition, jointVelocity);

    time = cellfun(@(x) double(x.header.stamp.sec) + 1e-9*double(x.header.stamp.nanosec) ...
        - 1e-9*double(bag.StartTime), msgs)';

    writematrix(jointPosition, strcat(csvPath, string(iTraj-10), '_jointPosition.txt' ))
    writematrix(position, strcat(csvPath, string(iTraj-10), '_eePosition.txt' ))
    writematrix(jointVelocity, strcat(csvPath, string(iTraj-10), '_jointVelocity.txt' ))
    writematrix(velocity, strcat(csvPath, string(iTraj-10), '_eeVelocity.txt' ))
    writematrix(time, strcat(csvPath, string(iTraj-10), '_time.txt' ))
% 
%     desiredJointVelocity = reshape(cell2mat({msgsStruct.effort}), 7, []);
%     desiredCartesianVelocity = pandaRobot.computeForwardVelocity(jointPosition, desiredJointVelocity);
    
    hold on;
    plot3(position(1, :), position(2, :), position(3, :)); axis equal

%     % Resample trajectories
%     trajectories(:, :, iTraj) = [interp1(time, position', linspace(time(1), time(end), nSamplePoint)), ...
%                                       interp1(time, velocity', linspace(time(1), time(end), nSamplePoint))]';
end

% saveDirectory = fullfile(filepath, '..', 'matlab_exercises', 'practical_1', 'dataset', 'demonstration_dataset.mat');
% save(saveDirectory, 'trajectories');
% disp("Saved datset to " + saveDirectory);

%%
% desiredCartesianVelocity = linearDS(position);
% 
% desiredJointVelocity = nan(size(jointVelocity));
% for i=1:size(position, 2)
% 
%     jacobian = pandaRobot.robot.geometricJacobian(jointPosition(:,i), 'panda_link8');
%     jacobianInverse = jacobian' / (jacobian*jacobian' + 0.01*eye(6));
% 
%     desiredJointVelocity(:, i) = jacobianInverse*[desiredCartesianVelocity(:, i); zeros(3, 1)];
% 
% end



% nSkip = 300;
% quiver3(position(1, 1:nSkip:end), position(2, 1:nSkip:end), position(3, 1:nSkip:end), ...
%         velocity(1, 1:nSkip:end),velocity(2, 1:nSkip:end), velocity(3, 1:nSkip:end))
% 
% quiver3(position(1, 1:nSkip:end), position(2, 1:nSkip:end), position(3, 1:nSkip:end), ...
%         desiredCartesianVelocity(1, 1:nSkip:end),desiredCartesianVelocity(2, 1:nSkip:end), desiredCartesianVelocity(3, 1:nSkip:end))
% 
% 
% plot3(position(1, :), position(2, :), position(3, :))

%%
figure(1)

axesName = ["X", "Y", "Z"];
for i =1:3
    
    subplot(3,1,i); hold on;
    plot(time, velocity(i, :))
    plot(time, desiredCartesianVelocity(i, :))
    ylabel(axesName(i) +" axis")
    legend(["actual", "desired"])
end
sgtitle("Task velocity [m/s]")

figure(2)
for i =1:7
    
    subplot(7,1,i); hold on;
    plot(time, jointVelocity(i, :))
    plot(time, desiredJointVelocity(i, :))
    legend(["actual", "desired"])
end
sgtitle("Joint velocity [rad/s]")
