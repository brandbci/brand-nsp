if exist('is_recording') && is_recording;
cbmex('fileconfig','','',0, 'option', 'close');
pause(1);
cbmex('close');
fprintf('File recording stopped at %s\n', datestr(now));
is_recording = false;
end