# peptoid-tools
Python library for creating peptoid structures (bilayers, alternating polymers, carbon chains) in .pdb format and packing them (sheets, tubes)

### Assembling a peptoid sheet
First initialize a builder object and build a single peptoid (available res can be found in res_lib. We'll be looking at BTM_perp and BTM_par for the perpendicular and parallel arrangements

```builder = assembler.Builder()
builder.assemble('BTM_perp', 6, 'CTM', 6, 'bilayer')
```
