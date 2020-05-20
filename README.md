# peptoid-tools
Python library for creating peptoid structures (bilayers, alternating polymers, carbon chains) in .pdb format and packing them (sheets, tubes)

Import using `from peptoid_tools import assembler`

### Assembling a peptoid sheet
First initialize a builder object and build a single peptoid (available res can be found in res_lib. We'll be looking at BTM_perp and BTM_par for the perpendicular and parallel arrangements

```
builder = assembler.Builder()
builder.assemble('BTM_perp', 6, 'CTM', 6, 'bilayer')
```

Next we initialize a packer object that packs the peptoid into a sheet. The x_gen() method copies the peptoid in the x-dimension n times with a distance of d in between oligomers. Similarly, y_gen() copies the newly created x-dimension n times in the y direction - again with a distance of d between each copy. Distances are initialized to realistic values so to create a bigger sheet all you need to adjust are the number of repeats in x and y.

```
packer = assembler.Packer(builder)
packer.x_gen(n_repeats=5, d=18, ring_type='perpendicular')
packer.copy_in_dim(n_repeats=17, d=4.5)
```

The last thing we do is create a writer object and send it the packer object which contains the fully formed sheet as well as the filename we want to write to.

```
writer = assembler.PDBWriter()
writer.write_pack(packer, 'thic_perp_18.pdb')
```
