%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read DTD dataset imdb
% then write a data.csv for lua etc.
%
% buyuantb@163.com 2016/10/11
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

imdb = load('imdb.mat');
fid = fopen('data.csv','w');
fprintf(fid,'filename,labelname,label,set\n');
images = imdb.images;
meta = imdb.meta;

for i = 1:size(images.id,2)
    fprintf(fid, '%s,%s,%d,%d\n', images.name{i},meta.classes{images.class(i)},images.class(i),images.set(i));
end

fclose(fid);