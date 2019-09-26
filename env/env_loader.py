import numpy as np
from matplotlib import pyplot as plt
from .json_reader import read_json, write_json
from pyram.PyRAM import PyRAM
from copy import deepcopy
from pyat.pyat.env import SSPraw, HS, TopBndry, BotBndry, Bndry, Source, Pos, Dom, SSP, Beam, cInt
from pyat.pyat.readwrite import write_env, write_fieldflp
from swellex_helpers.CTD.read_ctd import parse_prn
import scipy

'''
Description:
class to retrieve environmental parameters for running models

Author: Hunter Akins
'''

class Env:
    """
    Hold information related to the environment for model runs
    """
    def __init__(self, z_ss, rp_ss, cw, z_sb, rp_sb, cb, rhob, attn, rbzb):
        """ 
        z_ss: Water sound speed profile depths (m), NumPy 1D array.
        rp_ss: Water sound speed profile update ranges (m), NumPy 1D array.
        cw: Water sound speed values (m/s),
            Numpy 2D array, dimensions z_ss.size by rp_ss.size.
        z_sb: Seabed parameter profile depths (m), NumPy 1D array.
        rp_sb: Seabed parameter update ranges (m), NumPy 1D array.
        cb: Seabed sound speed values (m/s),
            NumPy 2D array, dimensions z_sb.size by rp_sb.size.
        rhob: Seabed density values (g/cm3), same dimensions as cb
        attn: Seabed attenuation values (dB/wavelength), same dimensions as cb
        rbzb: Bathymetry (m), Numpy 2D array with columns of ranges and depths
        """ 
        self.z_ss = z_ss
        self.rp_ss = rp_ss
        self.cw = cw
        self.z_sb = z_sb
        self.rp_sb = rp_sb
        self.cb = cb
        self.rhob = rhob
        self.attn = attn
        self.rbzb = rbzb
        self.env_dict = self.init_dict()

    def init_dict(self):
        env_dict = dict(z_ss = self.z_ss, 
                            rp_ss = self.rp_ss,
                            cw  = self.cw,
                            z_sb = self.z_sb,
                            rp_sb= self.rp_sb,
                            cb   = self.cb,
                            rhob = self.rhob,
                            rbzb = self.rbzb,
                            attn = self.attn)
        return env_dict

    def add_field_params(self, dz, zmax, dr, rmax, zmplt=None):
        """
        dz : Depth step (m), float
        zmax : Maximum depth (m), float
        dr : Range step (m), float
        rmax : Maximum range of model (m), float
        zmplt : Maximum output depth (m) , float
        """
        self.dz = dz
        self.zmax = zmax
        self.dr = dr
        self.rmax = rmax
        if zmplt is None:
            zmplt = np.max(self.rbzb)
        self.zmplt = zmplt
        self.env_dict['dz'] = dz
        self.env_dict['zmax'] = zmax
        self.env_dict['dr'] = dr
        self.env_dict['rmax'] = rmax
        self.env_dict['zmplt'] = zmplt
        return

    def add_source_params(self, freq, zs, zr):
        """
        freq : Source frequency (Hz), float
        zs : Source depth (m), float
        zr : Receiver depth (m), float
        """
        self.freq = freq
        self.zs = zs
        self.zr = zr
        self.env_dict['freq'] = freq
        self.env_dict['zs'] = zs
        self.env_dict['zr'] = zr
        return

    def init_pyram(self):
        """
        Initialize a pyram object from the env info
        """
        #freq, zs, zr, z_ss, rp_ss, cw = self.env_dict['freq'], self.env_dict['zs'], self.env_dict['zr'], self.env_dict['z_ss'], self.env_dict['rp_ss'], self.env_dict['cw']  
        env_dict = deepcopy(self.env_dict)
        pyram = PyRAM(env_dict['freq'], env_dict['zs'], env_dict['zr'],
                      env_dict['z_ss'], env_dict['rp_ss'], env_dict['cw'],
                      env_dict['z_sb'], env_dict['rp_sb'], env_dict['cb'],
                      env_dict['rhob'], env_dict['attn'], env_dict['rbzb'],
                      rmax=env_dict['rmax'], dr=env_dict['dr'],
                      dz=env_dict['dz'], zmplt=env_dict['zmplt'])
        return pyram

    def exp_to_json(self, name):
        write_json(name, self.env_dict)

    def pop_Pos(self, zr_flag=True):
        """
        Initialize a Pos object with field and source params
        Creates an attribute, pos, with the aforementioned Pos object
        returns a Pos object, pos
        """
        sd = self.zs
        s = Source(sd)
        X = np.arange(self.dr, self.rmax+self.dr, self.dr) * 1e-3
        if zr_flag == True:
            zr = self.zr
            if type(zr) == int:
                zr = [zr]
            r = Dom(X, zr)
        else:
            Z = np.arange(self.dz, self.zmax+self.dz, self.dz)
            r = Dom(X, Z)
        pos = Pos(s,r)
        pos.s.depth	= [sd]
        pos.Nsd = 1
        pos.Nrd = len([self.zr])
        self.pos = pos
        return pos

    def write_at_file(self, name, zr_flag=True):
        """
        Export env info to an .env file for kraken
        pyram doesn't have a halfspace, whereas kraken does
        Therefore, take the bottom layer (that's designed to kill the pe field) and turn it into a halfspace
        """
        cw = self.cw[:,0]
        ssp1 = SSPraw(self.z_ss, cw, np.zeros(cw.shape), np.ones(cw.shape),np.zeros(cw.shape), np.zeros(cw.shape))
        ssps = [ssp1]
        for i in range(len(self.z_sb) -2):
            z_points = [self.z_sb[i], self.z_sb[i+1]]
            c_points = [self.cb[i,0], self.cb[i+1, 0]]
            attn = [self.attn[i,0], self.attn[i+1,0]]
            rhob = [self.rhob[i,0], self.rhob[i+1,0]]
            tmp_ssp = SSPraw(z_points, c_points, 
                             [0]*len(c_points),
                             rhob, attn, [0]*len(c_points))
            ssps.append(tmp_ssp)
        
        nmedia = len(self.z_sb) - 1 # doesn't include halfspace ?
        depths = [0] + list(self.z_sb[:-1]) # chop off bottommost point
        lam = 1500 / self.freq
        # 10 points per meters, 1 / lam wavelengths per meter
        N = [int(20 / lam * (np.max(x.z) - np.min(x.z))) for x in ssps] # point per meter
        sigma = [0]*(nmedia+1)
        ssp = SSP(ssps, depths, nmedia, 'A', N, sigma)
        # add  halfspace after last layer
        hs = HS(self.cb[-1,0], 0, self.rhob[-1,0], self.attn[-1,0], 0)
#        hs = HS()
        top_bndry = TopBndry('CVW')
        bot_bndry = BotBndry('A', hs)
        bndry = Bndry(top_bndry, bot_bndry)
        pos = self.pop_Pos(zr_flag)
        cint = cInt(np.min(cw), self.cb[0][0])
        beam = None
        write_env(name, 'KRAKEN', 'Auto gen from Env object', self.freq, ssp, bndry, pos, beam, cint, self.rmax)
        return

    def write_flp(self, name, source_opt, zr_flag=True):
        pos = self.pop_Pos(zr_flag)
        write_fieldflp(name, source_opt, pos)
        return

         
class SwellexEnv(Env):
    """
    Inherit from Env, but add some method for swapping out ssp's conveniently using specific swellex EOFs and profiles
    Default Swellex environment has weirdly spaced depth points, so an interpolating scheme is nevessary to blend in other profiles from the CTD database
    """

    def change_cw(self, prn_file):
        """
        Read in a prn file and overwrite the Env ssp with the prn file ssp
        All prn files will have z points that are a subset of self.z
        Therefore interpolating on that segment is fine, and a blending scheme is used to 
        extrapolate them onto the lower ssp points of the default environment
        """
        z, ssp = parse_prn(prn_file)
        sspf = scipy.interpolate.interp1d(z, ssp)
        max_depth = np.max(z)
        lowest_ind = [i for i in range(len(self.z_ss)) if (self.z_ss[i] - max_depth) >0][0] - 1
        new_cw = sspf(self.z_ss[:lowest_ind])
        # blend them
        taper_len,taper_offset = 6, 4
        if taper_len+lowest_ind < self.z_ss.size:
            taper_vals = np.zeros(2*taper_len)
            taper_z = np.zeros(2*taper_len)
            bottom_vals = self.cw[lowest_ind:lowest_ind+taper_len,0] 
            top_vals = new_cw[-taper_len-taper_offset:-taper_offset]
            taper_vals[:taper_len] = top_vals
            taper_vals[taper_len:] = bottom_vals
            taper_z[:taper_len] = z[-taper_len:]
            taper_z[taper_len:] = self.z_ss[lowest_ind:lowest_ind+taper_len]
            taperf = scipy.interpolate.interp1d(taper_z, taper_vals)
            new_vals = taperf(z[-taper_len:])
            new_cw[-taper_len:] = new_vals
        # smoothly taper to that point
        self.cw[:new_cw.size,0] = new_cw
        
        
 
def env_from_json(name):
    env_dict = read_json(name)
    env_from_dict(env_dict)
    return env
    
def env_from_dict(env_dict, swellex=False):
    if swellex==True:
        env = SwellexEnv(env_dict['z_ss'], env_dict['rp_ss'], env_dict['cw'], env_dict['z_sb'], env_dict['rp_sb'], env_dict['cb'], env_dict['rhob'], env_dict['attn'], env_dict['rbzb'])
    else:
        env = Env(env_dict['z_ss'], env_dict['rp_ss'], env_dict['cw'], env_dict['z_sb'], env_dict['rp_sb'], env_dict['cb'], env_dict['rhob'], env_dict['attn'], env_dict['rbzb'])
    return env
    
class EnvFactory:
    """
    Registers environment builders  for various environments
    A builder is identified with a key
    Builder implements the __call__ method and returns an Env object
    """
    def __init__(self):
        self.builders = {}

    def register_builder(self, key, builder):
        self.builders[key] = builder

    def create(self, key, **kwargs):
        """ 
        Call the builder associated with key on kwargs
        """
        builder = self.builders[key]
        if not builder:
            raise KeyError(key)
        return builder(**kwargs)
        
        
    
