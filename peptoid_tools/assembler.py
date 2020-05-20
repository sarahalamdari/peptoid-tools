import os
import math
import pkg_resources
import numpy as np

RES_LIB = pkg_resources.resource_filename('peptoid-tools.peptoid_tools', 'res_lib')

def vec_align(a, b):
    """
    Align two vectors

    Parameters
    ----------

    """
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = a.dot(b)
    I = np.identity(3)
    v_skew = np.zeros((3,3))
    v_skew[0,0] = 0
    v_skew[1,0] = v[2]
    v_skew[2,0] = -v[1]
    v_skew[0,1] = -v[2]
    v_skew[1,1] = 0
    v_skew[2,1] = v[0]
    v_skew[0,2] = v[1]
    v_skew[1,2] = -v[0]
    v_skew[2,2] = 0
    R = I + v_skew + v_skew@v_skew*(1/(1+c))
    return R

def get_rot_mat(axis, angle):
    rot_mat = np.zeros((3,3))
    rot_mat[0,0] = math.cos(angle)+axis[0]**2*(1-math.cos(angle))
    rot_mat[1,0] = axis[0]*axis[1]*(1-math.cos(angle))+axis[2]*math.sin(angle)
    rot_mat[2,0] = axis[0]*axis[2]*(1-math.cos(angle))-axis[1]*math.sin(angle)
    rot_mat[0,1] = axis[0]*axis[1]*(1-math.cos(angle))-axis[2]*math.sin(angle)
    rot_mat[1,1] = math.cos(angle)+axis[1]**2*(1-math.cos(angle))
    rot_mat[2,1] = axis[1]*axis[2]*(1-math.cos(angle))+axis[0]*math.sin(angle)
    rot_mat[0,2] = axis[0]*axis[2]*(1-math.cos(angle))+axis[1]*math.sin(angle)
    rot_mat[1,2] = axis[1]*axis[2]*(1-math.cos(angle))-axis[0]*math.sin(angle)
    rot_mat[2,2] = math.cos(angle)+axis[2]**2*(1-math.cos(angle))
    return rot_mat

class PDBWriter:
    def __init__(self):
        self.header = "CRYST1   40.000   40.000   40.000  90.00  90.00  90.00 P 1           1"+'\n'
        self.tail = "END"

    def write_line(self, atomid, atomtype, resname, resid, position):
        line = f"ATOM{atomid:7d}  {atomtype:<3s}{resname:>4s}{resid:6d}{position[0]:12.3f}{position[1]:8.3f}{position[2]:8.3f}  1.00  0.00"+'\n'
        return line

    def write_res(self, res, fo, atomid=1, resid=1, write_out=True):
        atoms = res.atoms
        if write_out:
            fo = open(fo, 'w')
            fo.write(self.header)
        if '_' in res.name:
            resname = res.name.split('_')[0]
        else:
            resname = res.name
        for i, atom in enumerate(atoms):
            id = i+atomid
            atomtype = atom
            position = res.positions[:,i]
            line = self.write_line(id, atomtype, resname, resid, position)
            fo.write(line)
        if write_out:
            fo.write(self.tail)

    def write_build(self, builder, fo, atomid=1, resid=1, write_out=True):
        if write_out:
            fo = open(fo, 'w')
            fo.write(self.header)
        for i, res in enumerate(builder.chain):
            self.write_res(res, fo, atomid, resid=i+resid, write_out=False)
            atomid += len(res.atoms)
        if write_out:
            fo.write(self.tail)

    def write_pack(self, packer, fo, write_out=True):
        atomid = 1
        num_res = len(packer.build.chain)
        if write_out:
            fo = open(fo, 'w')
            fo.write(self.header)
        for i, oligomer in enumerate(packer.oligomers):
            for j, res in enumerate(oligomer.chain):
                resid = i*num_res+(j+1)
                self.write_res(res, fo, atomid, resid=resid, write_out=False)
                atomid += len(res.atoms)
        if write_out:
            fo.write(self.tail)

class Packer:
    def __init__(self, build):
        self.build = build
        self.oligomers = [build]

    def make_new_copy(self, oligomer='default'):
        if oligomer == 'default':
            copy = self.build.__copy__()
        else:
            copy = oligomer.__copy__()
        self.oligomers.append(copy)

    def backbone_align(self, oligomer1, oligomer2, dim_ignore=None):
        for res in oligomer1.chain:
            if res.type == 'hydrophobic':
                last_res = res
        for i, atom in enumerate(last_res.atoms):
            if atom == 'NL':
                align_pos = last_res.positions[:,i]
        first_res = oligomer2.chain[0]
        for i, atom in enumerate(first_res.atoms):
            if atom == 'NR':
                move_pos = first_res.positions[:,i]
        x_move = align_pos[0] - move_pos[0]
        y_move = align_pos[1] - move_pos[1]
        z_move = align_pos[2] - move_pos[2]
        trltn = [x_move, y_move, z_move]
        if dim_ignore != None:
            trltn[dim_ignore] = 0
        oligomer2.trans(trltn)

    def x_gen(self, n_repeats, d=18, ring_type='parallel'):
        for i in range(n_repeats):
            self.make_new_copy(self.oligomers[i])
            self.oligomers[-1].flip_x(np.pi)
            if ring_type == 'parallel':
                self.oligomers[-1].flip_z(np.pi)
            self.oligomers[-1].trans([d,0,0])
            self.backbone_align(self.oligomers[-2], self.oligomers[-1], dim_ignore=0)

    def y_gen(self, n_repeats, d=4.5, dim=1):
        bulk_oligomers = [self.oligomers]
        trltn = np.zeros((3))
        trltn[dim] = d

        for i in range(n_repeats):
            new_oligomers = []
            for oligomer in bulk_oligomers[-1]:
                new_oligomers.append(oligomer.__copy__())
            for oligomer in new_oligomers:
                oligomer.trans(trltn)
                self.oligomers.append(oligomer)
            bulk_oligomers.append(new_oligomers)

class Builder:
    def __init__(self):
        self.chain = []
        self.orientation_up = True
        self.res_types = [f[:-4] for f in os.listdir(RES_LIB) if f[-4:] == '.pdb']

    def __copy__(self):
        builder_copy = Builder()
        for res in self.chain:
            builder_copy.chain.append(res.__copy__())
        builder_copy.orientation_up = self.orientation_up
        builder_copy.res_types = self.res_types
        return builder_copy

    def chain_init(self, resname):
        res1 = PeptoidResidue(resname)
        self.chain.append(res1)
        self.orientation_up = not self.orientation_up

    def carbon_chain_init(self):
        res1 = CarbonResidue('PCH')
        res1.name = 'PCT'
        self.chain.append(res1)

    def repeat(self, n=1):
        repeat_count = 1
        res1 = self.chain[-1]
        res2 = res1.__copy__()
        # Align on res axis
        res1.get_backbone_vectors(inplace=True)
        res2.get_backbone_vectors(inplace=True)
        R = vec_align(res1.backbone_vectors[1], res2.backbone_vectors[1])
        res2.rotate(math.pi)
        self.align_nl(res1, res2)

        res2_begin = res1.positions[:,res1.atoms.index('CLP')] - res1.backbone_vectors[1]
        trltn = res2_begin - res2.positions[:,res2.atoms.index('CA')]
        res2.translate(trltn)

        self.chain.append(res2)
        self.orientation_up = not self.orientation_up
        if repeat_count < n-1:
            self.repeat(n-1)

    def alternate(self, restypes, n=1):
        repeat_count = 1
        res1 = self.chain[-1]
        idx1 = restypes.index(res1.name)
        if idx1 == 0:
            type_to_add = restypes[1]
        elif idx1 == 1:
            type_to_add = restypes[0]
        res2 = PeptoidResidue(type_to_add)

        # Set origin
        origin_trltn = -self.chain[0].positions[:,self.chain[0].atoms.index('CA')]
        for res in self.chain:
            res.translate(origin_trltn)
        origin_trltn_res = -res2.positions[:,res2.atoms.index('CA')]
        res2.translate(origin_trltn_res)

        # Align on z-axis
        v_chain = self.chain[0].positions[:,self.chain[0].atoms.index('CLP')] - self.chain[0].positions[:,self.chain[0].atoms.index('CA')]
        v_chain = v_chain / np.linalg.norm(v_chain)

        v_res = res2.positions[:,res2.atoms.index('CLP')] - res2.positions[:,res2.atoms.index('CA')]
        v_res = v_res / np.linalg.norm(v_res)

        a_chain = v_chain
        a_res = v_res
        b = np.array([0, 0, 1])
        R_chain = vec_align(a_chain,b)
        R_res = vec_align(a_res,b)
        for res in self.chain:
            new_pos = R_chain.dot(res.positions)
            res.positions = new_pos
        new_pos = R_res.dot(res2.positions)
        res2.positions = new_pos
        if self.orientation_up:
            v_x = ((res2.positions[:,res2.atoms.index('CLP')] - res2.positions[:,res2.atoms.index('CA')]) / 2) - res2.positions[:,res2.atoms.index('NL')]
        else:
            v_x = res2.positions[:,res2.atoms.index('NL')] - ((res2.positions[:,res2.atoms.index('CLP')] - res2.positions[:,res2.atoms.index('CA')]) / 2)
        v_x = v_x / np.linalg.norm(v_x)
        R_x = vec_align(v_x, np.array([1, 0, 0]))
        new_pos = R_x.dot(res2.positions)
        res2.positions = new_pos

        self.align_nl(res1, res2)
        res1.get_backbone_vectors(inplace=True)
        res2.get_backbone_vectors(inplace=True)

        res2_begin = res1.positions[:,res1.atoms.index('CLP')] - res1.backbone_vectors[1]
        trltn = res2_begin - res2.positions[:,res2.atoms.index('CA')]
        res2.translate(trltn)

        self.chain.append(res2)
        self.orientation_up = not self.orientation_up
        if repeat_count < n-1:
            self.alternate(restypes, n-1)

    def create_carbon_chain(self, n=1):
        repeat_count = 1
        res1 = self.chain[-1]
        res2 = res1.__copy__()
        res2.name = 'PCH'
        res2.rotate(math.pi)

        if self.orientation_up:
            vec = res1.down_vec
        else:
            vec = res1.up_vec
        res2_begin = res1.positions[:,0] + vec
        trltn = res2_begin - res2.positions[:,0]
        res2.translate(trltn)

        self.chain.append(res2)
        self.orientation_up = not self.orientation_up
        res2.up_status = not res2.up_status
        if repeat_count < n-1:
            self.create_carbon_chain(n-1)

    def align_nl(self, res1, res2):
        nl_pos1 = res1.positions[:,res1.atoms.index('NL')]
        nl_pos2 = res2.positions[:,res2.atoms.index('NL')]
        trltn = nl_pos1 - nl_pos2
        res2.translate(trltn)

    def chain_align(self, chain1, chain2):
        ### Set origin (0,0,0)
        trltn1 = -chain1[-1].positions[:,0]
        for res in chain1:
            res.translate(trltn1)
        trltn2 = -chain2[0].positions[:,chain2[0].atoms.index('CA')]
        for res in chain2:
            res.translate(trltn2)

        ### Align chains on z-axis
        v2 = chain2[0].positions[:,chain2[0].atoms.index('CLP')] - chain2[0].positions[:,chain2[0].atoms.index('CA')]
        v2 = v2 / np.linalg.norm(v2)

        a = v2
        b = np.array([0, 0, 1])
        R = vec_align(a,b)
        for res in chain2:
            new_pos = R.dot(res.positions)
            res.positions = new_pos

        ### Align chain backbone vertically
        if chain1[-1].up_status == False:
            b = np.array([-1, 0, 0])
            trans_v = chain1[-1].up_vec
        else:
            b = np.array([1, 0, 0])
            trans_v = chain1[-1].down_vec

        a = chain2[0].positions[:,chain2[0].atoms.index('CB')] - chain2[0].positions[:,chain2[0].atoms.index('NL')]
        a = a / np.linalg.norm(a)
        R = vec_align(a,b)
        for res in chain2:
            new_pos = R.dot(res.positions)
            res.positions = new_pos

        ### Shift carbon chain to end of peptide chain
        carb_start = chain2[-1].positions[:,chain2[0].atoms.index('CLP')] + trans_v
        trltn = carb_start - chain1[0].positions[:,0]
        for res in chain1:
            res.translate(trltn)

    def assemble(self, restype1, n1, restype2='STM', n2=0, chain_type='bilayer', n_cap_type='amine', c_cap_type='amine'):
        if restype1 not in self.res_types or restype2 not in self.res_types:
            print("Please select a valid residue (Builder.res_types)")
            return 0
        if chain_type == 'bilayer':
            self.chain_init(restype1)
            self.repeat(n1)
            if n2 > 0:
                self.alternate([restype1, restype2])
                self.repeat(n2)
            self.chain_type = 'bilayer'
        elif chain_type == 'alternating':
            self.chain_init(restype1)
            self.alternate([restype1, restype2], n1)
            self.chain_type = 'alternating'
        elif chain_type == 'carbon':
            self.chain_init(restype1)
            self.repeat(n1)
            self.carbon_chain_init()
            # self.alternate([restype1, 'PCH'])
            self.create_carbon_chain(n2)
            self.chain_type = 'carbon'
            chain1 = self.chain[n1:]
            chain2 = self.chain[:n1]
            self.chain_align(chain1, chain2)
            self.chain[-1].make_carbon_cap(c_cap_type)
            self.chain[0].make_n_cap(n_cap_type)
            return 0
        self.chain[-1].make_c_cap(c_cap_type)
        self.chain[0].make_n_cap(n_cap_type)

    def trans(self, trltn):
        for res in self.chain:
            res.translate(trltn)

    def flip_x(self, angle):
        rot_mat = np.zeros((3,3))
        rot_mat[0,0] = 1
        rot_mat[1,0] = 0
        rot_mat[2,0] = 0
        rot_mat[0,1] = 0
        rot_mat[1,1] = math.cos(angle)
        rot_mat[2,1] = -math.sin(angle)
        rot_mat[0,2] = 0
        rot_mat[1,2] = math.sin(angle)
        rot_mat[2,2] = math.cos(angle)
        for res in self.chain:
            new_pos = rot_mat.dot(res.positions)
            res.positions = new_pos

    def flip_z(self, angle):
        # Move to origin
        origin_trltn = -self.chain[0].positions[:,0]
        self.trans(origin_trltn)
        axis = [0, 0, 1]
        rot_mat = get_rot_mat(axis, angle)
        for res in self.chain:
            new_pos = rot_mat.dot(res.positions)
            res.positions = new_pos
        self.trans(-origin_trltn)

    def clear(self):
        self.chain = []
        self.orientation_up = True

class Residue:
    def __init__(self, name):
        self.name = name
        self.file_name = self.name
        self.pdb_path = RES_LIB+"/"+name+".pdb"
        self.atoms = self.get_atom_names()
        self.positions = self.get_atom_positions()

    def __copy__(self):
        new_res = Residue(self.name)
        new_res.positions = self.positions
        return new_res

    def get_atom_names(self):
        atom_list = []
        with open(self.pdb_path, 'r') as file:
            for line in file:
                line = line.split()
                try:
                    if line[0] == 'ATOM':
                        atom_list.append(line[2])
                except IndexError:
                    pass
        return atom_list

    def get_atom_positions(self):
        pos_array = np.zeros((3,len(self.atoms)))
        i = 0
        with open(self.pdb_path, 'r') as file:
            for line in file:
                line = line.split()
                try:
                    if line[0] == 'ATOM':
                        pos_array[:,i] = [float(line[6]), float(line[7]), float(line[8])]
                        i += 1
                except IndexError:
                    pass
        return pos_array

    def translate(self, trltn, cur_pos=True):
        if isinstance(cur_pos, list):
            pass
        elif cur_pos == True:
            cur_pos = self.positions
        trltn = np.array(trltn)
        new_pos = cur_pos+trltn[:,None]
        self.positions = new_pos

class CarbonResidue(Residue):
    def __init__(self, name):
        Residue.__init__(self, name)
        self.type = 'carbon'
        self.down_vec = np.array([-1, 0, 1])
        self.up_vec = np.array([1, 0, 1])
        self.up_status = True

    def __copy__(self):
        new_res = CarbonResidue('PCH')
        if self.name[0] == 'C':
            new_res.make_carbon_cap('methyl')
        elif self.name[0] == 'N':
            new_res.make_carbon_cap('amine')
        elif self.name == 'PCT':
            new_res.name = 'PCT'
        new_res.positions = self.positions
        new_res.up_status = self.up_status
        return new_res

    def rotate(self, angle):
        rot_mat = np.zeros((3,3))
        rot_mat[0,0] = math.cos(angle)
        rot_mat[1,0] = -math.sin(angle)
        rot_mat[2,0] = 0
        rot_mat[0,1] = math.sin(angle)
        rot_mat[1,1] = math.cos(angle)
        rot_mat[2,1] = 0
        rot_mat[0,2] = 0
        rot_mat[1,2] = 0
        rot_mat[2,2] = 1
        new_pos = rot_mat.dot(self.positions)
        trltn = self.positions[:,0] - new_pos[:,0]
        self.translate(trltn, list(new_pos))

    def make_carbon_cap(self, c_cap_type):
        if c_cap_type == 'amine':
            self.atoms[0] = 'NT'
            self.atoms[1] = 'HN1'
            self.atoms[2] = 'HN2'
            self.name = 'NCH'

        elif c_cap_type == 'methyl':
            vec = -self.up_vec
            vec = vec / np.linalg.norm(vec)
            hp3_pos = self.positions[:,0] - vec

            self.atoms.append('HP3')
            new_pos = np.zeros((3,len(self.atoms)))
            new_pos[:,:3] = self.positions
            new_pos[:,3] = hp3_pos
            self.positions = new_pos
            self.name = 'CCH'

class PeptoidResidue(Residue):
    def __init__(self, name):
        Residue.__init__(self, name)
        if self.name[0] == 'B':
            self.type = 'hydrophobic'
        elif self.name[0] == 'C':
            self.type = 'polar'
        self.backbone_atoms = ['CLP','NL','CA']
        self.backbone_idxs, self.backbone_vectors = self.get_backbone_vectors()
        self.move_ol()

    def __copy__(self):
        new_res = PeptoidResidue(self.file_name)
        new_res.backbone_idxs, new_res.backbone_vectors = new_res.get_backbone_vectors()
        if self.name[-2:] == 'NR':
            new_res.make_n_cap('amine')
        elif self.name[-2:] == 'CR':
            new_res.make_c_cap('amine')
        new_res.positions = self.positions
        return new_res

    def get_backbone_vectors(self, inplace=False):
        bb_idxs = []
        vec_dict = {}
        for i, atom in enumerate(self.atoms):
            if atom in self.backbone_atoms:
                bb_idxs.append(i)
        vec = self.positions[:,bb_idxs[2]] - self.positions[:,bb_idxs[0]]
        v1 = self.positions[:,bb_idxs[2]] - self.positions[:,bb_idxs[1]]
        vec_len = np.linalg.norm(vec)
        v_norm = vec / vec_len
        if inplace:
            self.backbone_vectors = [v_norm, v1]
        else:
            return bb_idxs, [v_norm, v1]

    def move_ol(self):
        clp_index = self.atoms.index('CLP')
        clp_pos = self.positions[:,clp_index]
        ol_index = self.atoms.index('OL')
        ol_pos = self.positions[:,ol_index]
        x, y, _ = clp_pos - ol_pos
        angle = -math.pi / 1.45
        new_x = -y*math.sin(angle) + x*math.cos(angle)
        new_y = y*math.cos(angle) + x*math.sin(angle)
        new_ol_pos = [clp_pos[0] - new_x, clp_pos[1] - new_y, clp_pos[2]]

        self.positions[:,ol_index] = new_ol_pos

    def rotate(self, angle):
        bb_v = self.backbone_vectors[0]
        rot_mat = np.zeros((3,3))
        rot_mat[0,0] = math.cos(angle)+bb_v[0]**2*(1-math.cos(angle))
        rot_mat[1,0] = bb_v[0]*bb_v[1]*(1-math.cos(angle))+bb_v[2]*math.sin(angle)
        rot_mat[2,0] = bb_v[0]*bb_v[2]*(1-math.cos(angle))-bb_v[1]*math.sin(angle)
        rot_mat[0,1] = bb_v[0]*bb_v[1]*(1-math.cos(angle))-bb_v[2]*math.sin(angle)
        rot_mat[1,1] = math.cos(angle)+bb_v[1]**2*(1-math.cos(angle))
        rot_mat[2,1] = bb_v[1]*bb_v[2]*(1-math.cos(angle))+bb_v[0]*math.sin(angle)
        rot_mat[0,2] = bb_v[0]*bb_v[2]*(1-math.cos(angle))+bb_v[1]*math.sin(angle)
        rot_mat[1,2] = bb_v[1]*bb_v[2]*(1-math.cos(angle))-bb_v[0]*math.sin(angle)
        rot_mat[2,2] = math.cos(angle)+bb_v[2]**2*(1-math.cos(angle))
        new_pos = rot_mat.dot(self.positions)
        trltn = self.positions[:,0] - new_pos[:,0]
        self.translate(trltn, list(new_pos))

    def make_c_cap(self, cap_type):
        if cap_type == 'methyl':
            rand_v = np.random.rand(1,3)
            perp_v = np.cross(-self.backbone_vectors[1], rand_v)
            perp_v = perp_v / np.linalg.norm(perp_v)

            cl_pos = -self.backbone_vectors[1]+self.positions[:,self.atoms.index('CLP')]
            hl1_pos = cl_pos + perp_v
            hl2_pos = cl_pos - perp_v
            hl3_v = cl_pos - self.positions[:,self.atoms.index('CLP')]
            hl3_v = hl3_v / np.linalg.norm(hl3_v)
            hl3_pos = cl_pos + hl3_v
            ch3_pos = [cl_pos, hl1_pos, hl2_pos, hl3_pos]

            new_atoms = ['CL','HL1','HL2','HL3']
            new_atoms.reverse()
            for atom in new_atoms:
                self.atoms.insert(0, atom)

            new_pos = np.zeros((3,len(self.atoms)))
            for i, pos in enumerate(ch3_pos):
                new_pos[:,i] = pos
            new_pos[:,4:] = self.positions
            self.positions = new_pos
            self.name = self.name[:-1]+'N'

        elif cap_type == 'amine':
            hn1_pos = self.positions[:,0]
            self.atoms.pop(0)
            self.atoms.pop(0)
            self.atoms.insert(1, 'HN1')

            new_pos = np.zeros((3, len(self.atoms)))
            old_idx = 2
            for i in range(new_pos.shape[1]):
                if i == 1:
                    new_pos[:,i] = hn1_pos
                else:
                    new_pos[:,i] = self.positions[:,old_idx]
                    old_idx += 1

            self.positions = new_pos
            self.name = self.name[0]+'CR'

    def make_n_cap(self, cap_type):
        if cap_type == 'methyl':
            init_idxs = []
            for i, atom in enumerate(self.atoms):
                if atom in ['CA','HA1','HA2']:
                    try:
                        new_atom = atom[0] + 'R' + atom[2]
                    except IndexError:
                        new_atom = atom[0] + 'R'
                    self.atoms[i] = new_atom
                    split_idx = i+1
            h_v = self.backbone_vectors[1] / np.linalg.norm(self.backbone_vectors[1])
            h_pos = self.positions[:,self.atoms.index('CR')]+h_v
            self.atoms.insert(split_idx, 'HR3')
            new_pos = np.zeros((3,len(self.atoms)))
            new_pos[:,:split_idx] = self.positions[:,:split_idx]
            new_pos[:,split_idx] = h_pos
            new_pos[:,split_idx+1:] = self.positions[:,split_idx:]
            self.positions = new_pos
            self.name = self.name[:-1]+'C'


        elif cap_type == 'amine':
            self.atoms[0] = 'CLP'
            self.atoms[1] = 'OL'
            self.atoms[2] = 'NR'
            self.atoms[3] = 'CA1'

            C_to_N_vec = self.positions[:,2] - self.positions[:,0]
            C_to_O_vec = self.positions[:,0] - self.positions[:,1]
            N_to_C_vec = self.positions[:,3] - self.positions[:,2]
            N_to_H1_vec = self.positions[:,4] - self.positions[:,3]
            N_to_H2_vec = self.positions[:,5] - self.positions[:,3]

            cp2_pos = self.positions[:,3] + C_to_N_vec
            ol2_pos = cp2_pos + C_to_O_vec
            nt_pos = cp2_pos + N_to_C_vec
            hn1_pos = nt_pos + N_to_H1_vec
            hn2_pos = nt_pos + N_to_H2_vec

            new_atoms = ['CP2', 'OL2', 'NT', 'HN1', 'HN2']
            new_pos = [cp2_pos, ol2_pos, nt_pos, hn1_pos, hn2_pos]
            new_pos_arr = np.zeros((3,len(new_pos)))
            idx = 0
            for atom, pos in zip(new_atoms,new_pos):
                self.atoms.append(atom)
                new_pos_arr[:,idx] = pos
                idx += 1
            self.positions = np.concatenate((self.positions, new_pos_arr),axis=1)
            self.name = self.name[0]+'NR'
