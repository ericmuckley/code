i = 0;
while 1
    line = fgetl(position);
    sm = strmatch('# of Points',line);
    if ~isempty(sm)
        regex = '# of Points(.*)';
        [match tokens] = regexp(line,regex,'match','tokens');
        npoints = str2num(tokens{1}{1});
    elseif strcmp(line,'R.Time	Intensity')
        break
    i = i + 1
    end
end
