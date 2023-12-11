if exist('is_recording') && is_recording;
cbmex('fileconfig','','',0, 'option', 'close', 'instance', 1);
pause(1);
cbmex('close', 'instance', 1);
fprintf('File recording stopped at %s\n', datestr(now));
is_recording = false;
end