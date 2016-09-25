bllc = model:get(4)
llc = model:get(5)
allc = model:get(6)

bencode = bllc.output
aencode = llc.output

print(bencode)
print(aencode)
singlellc = nn.LLC(llc.B)


