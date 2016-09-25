bllc = model:get(4)
llc = model:get(5)
allc = model:get(6)

bencode = bllc.output
aencode = llc.output

--print "================== before encode =================="
--print(bencode)
print "================= after encode =================="
print(aencode)
singlellc = nn.LLC(llc.B)
output = torch.zeros(aencode:size(1),bencode:size(2),bencode:size(3))

print "================== single LLC =================="
print "================== single LLC =================="
print "================== single LLC =================="
for i=1,bencode:size(2) do
   for j=1,bencode:size(3) do
      --print "================== before encode =================="
      --print(bencode[{{},i,j}])
      output[{{},i,j}] = singlellc:forward(bencode[{{},i,j}])
      print "================= after encode =================="
      print(output[{{},i,j}])
   end
end
