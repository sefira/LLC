bllc = model:get(4)
llc = model:get(5)
allc = model:get(6)

bencode = bllc.output
aencode = llc.output

isprintbeforeencode = false
isprintafterencode = false
isprintgradinput = true
print "################# LLCNet ##################"
print "################# LLCNet ##################"
print "################# LLCNet ##################"
if isprintbeforeencode then
   print "================== before encode =================="
   print(bencode)
end
if isprintafterencode then 
   print "================= after encode =================="
   print(aencode)
end
if isprintgradinput then
   print "================= after gradinput =================="
   print(llc.gradInput)
end
singlellc = nn.LLC(llc.B)
output = torch.zeros(aencode:size(1),bencode:size(2),bencode:size(3))
gradInput = torch.zeros(bencode:size(1),bencode:size(2),bencode:size(3))

print "################# single LLC ##################"
print "################# single LLC ##################"
print "################# single LLC ##################"
for i=1,bencode:size(2) do
   for j=1,bencode:size(3) do
      if isprintbeforeencode then
         print "================== before encode =================="
         print(bencode[{{},i,j}])
      end
      output[{{},i,j}] = singlellc:forward(bencode[{{},i,j}])
      if isprintafterencode then
         print "================= after encode =================="
         print(output[{{},i,j}])
      end
      if isprintgradinput then
         gradInput[{{},i,j}] = singlellc:backward(bencode[{{},i,j}],allc.gradInput[{{},i,j}])
         print "================= after gradinput =================="
         print(gradInput[{{},i,j}])
      end
   end
end
