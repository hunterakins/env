import numpy as np
from matplotlib import pyplot as plt
from .json_reader import read_json, write_json
from pyram.pyram.PyRAM import PyRAM
from copy import deepcopy
from pyat.pyat.env import SSPraw, HS, TopBndry, BotBndry, Bndry, Source, Pos, Dom, SSP, Beam, cInt
from pyat.pyat.readwrite import write_env, write_fieldflp, read_shd, write_bathy, write_ssp, read_modes
from swellex.CTD.read_ctd import parse_prn
from scipy.interpolate import interp1d
from env.env.s5_helpers import s5_approach_D, get_range_correction
from pandas import read_feather
import scipy
import feather
from os import system

'''
Description:
class to retrieve environmental parameters for running models

Author: Hunter Akins
'''

def get_custom_r_modal_field(modes, r, zs, zr, tilt_angle=0):
    """
    Given modal object, range grid r, source depth(s) zs 
    and receiver depths zr (corresponding to the modes obj)
    Compute the field at the grid (zr X r)
    Input
    modes - Modes obj (pyat)
    r - numpy 1d array
        grid of range positions at which to evaluate the field
        IN METERS
    zs - np array
        source depths (compute for each depth)
    zr - numpy 1d array of floats (or possibly ints)
        receiver depths
    """
    if type(zs) == int or type(zs) == float:
        zs = np.array([zs])
    r_corr = get_range_correction(zs, tilt_angle)
    p = np.zeros((len(zs), len(zr), len(r)), dtype=np.complex128)
    for index, source_depth in enumerate(zs):
        krs = modes.k
        phi = modes.get_receiver_modes(zr)
        strength = modes.get_source_strength(source_depth)
        modal_matrix = strength*phi
        r_mat= np.outer(krs, r-r_corr[index])
        """
        Note...kraken c has the attenuation as a negative
        imaginary part to k
        The implied convention is exp(-i k r) transform...
        """
        range_dep = np.exp(complex(0,1)*r_mat.conj()) / np.sqrt(r_mat.real)
        source_p = modal_matrix@range_dep
        source_p *= -np.exp(complex(0, 1)*np.pi/4)
        source_p /= np.sqrt(8*np.pi)
        source_p = source_p.conj()
        p[index, :,:] = source_p
    return p

def get_sea_surface(cw):
    """
    Generate a little sine wave surface for 
    making graphics """
    min_c = np.min(cw)
    max_c = np.max(cw)
    crange = max_c - min_c
    lam = crange / 5
    x = np.linspace(min_c, max_c, 1000)
    y = np.sin(2*np.pi*x/lam)
    return x, y

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

    def pop_Pos(self, zr_flag=True, zr_range_flag=True,custom_r=None):
        """
        Initialize a Pos object with field and source params
        Creates an attribute, pos, with the aforementioned Pos object
        returns a Pos object, pos
        if zr_flag is true, field is evaluated at receiver positions
        It seems that rmax and dr are given in meters, but I convert them
        to km in Pos (since the models run on km)
        """
        sd = self.zs
        s = Source(sd)
        X = np.arange(self.dr, self.rmax+self.dr, self.dr) * 1e-3
        if zr_flag == True:
            zr = self.zr
            if type(zr) == int:
                zr = np.array([zr])
            if zr_range_flag == True:
                X = np.array([X[-1]]) # only compute last range pos
            r = Dom(X, zr)
        else:
            Z = np.arange(self.dz, self.zmax+self.dz, self.dz)
            r = Dom(X, Z)
        pos = Pos(s,r)
        if (type(pos.s.depth) == int) or (type(pos.s.depth) == float):
            pos.s.depth	= [sd]
        else:
            pos.s.depth = sd
        pos.Nsd = 1
        pos.Nrd = len([self.zr])
        if type(custom_r) != type(None):
            pos.r.range = custom_r*1e-3
        self.pos = pos
        return pos

    def write_env_file(self, name, model='kraken', zr_flag=True, zr_range_flag=True,beam=None, custom_r=None):
        """
        Input 
        name : string
            Full path of save location for env file
        model : string
            which model to use ('kraken', 'bellhop')
        zr_flag : bool
            If true evaluate the field at the zr positions. Otherwise the model will return the field at all the computed field points
        zr_range_flag : bool
            If true, only evaluates the field at rmax, the assumed range position. If false, it will evaluate the field at each range point
        beam : Beam object 
            For running bellhop, Beam sets parameters for ray trace
        Output:
            None. Writes a .env file to the path given in name. 
        Export env info to an .env file for kraken
        pyram doesn't have a halfspace, whereas kraken does
        Therefore, take the bottom layer (that's designed to kill the pe field) and turn it into a halfspace
        """
        cw_shape = self.cw.shape
        two_d_flag = False
        if len(cw_shape) == 2:
            if cw_shape[1] > 1:
                two_d_flag = True
        if two_d_flag == True:
            write_bathy(name, self.rbzb)
            write_ssp(name, self.cw, self.rp_ss)
        cw = self.cw[:,0] # only use first range prof for env file
        ssp1 = SSPraw(self.z_ss, cw, np.zeros(cw.shape), np.ones(cw.shape),np.zeros(cw.shape), np.zeros(cw.shape))
        ssps = [ssp1]
        nmedia = len(self.z_sb)//2 +1
        for i in range(nmedia-1): # ignore the water layer
            z_points = [self.z_sb[2*i], self.z_sb[2*i+1]]
            c_points = [self.cb[2*i,0], self.cb[2*i+1, 0]]
            attn = [self.attn[2*i,0], self.attn[2*i+1,0]]
            rhob = [self.rhob[2*i,0], self.rhob[2*i+1,0]]
            tmp_ssp = SSPraw(z_points, c_points, 
                             [0]*len(c_points),
                             rhob, attn, [0]*len(c_points))
            ssps.append(tmp_ssp)
        depths = [0] + list(set(list(self.z_sb)))
        depths.sort()
        if self.freq == 0:
            lam = 100 # dummy val
        else:
            lam = 1500 / self.freq
        # 10 points per meters, 1 / lam wavelengths per meter
        N = [int(20 / lam * (np.max(x.z) - np.min(x.z))) for x in ssps] # point per meter
        sigma = [0]*(nmedia+1)
        ssp = SSP(ssps, depths, nmedia, 'A', N, sigma)
        # add  halfspace after last layer
        hs = HS(self.cb[-1,0], 0, self.rhob[-1,0], self.attn[-1,0], 0)
#        hs = HS()
        if two_d_flag == True:
                top_bndry = TopBndry('QVW')
        else:
            top_bndry = TopBndry('CVW')
        bot_bndry = BotBndry('A', hs)
        bndry = Bndry(top_bndry, bot_bndry)
        pos = self.pop_Pos(zr_flag=zr_flag, zr_range_flag=zr_range_flag, custom_r=custom_r)
        cint = cInt(np.min(cw), self.cb[0][0])
        cint = cInt(1470, 1650)
        write_env(name, model, 'Auto gen from Env object', self.freq, ssp, bndry, pos, beam, cint, self.rmax)
        return
                
    def write_flp(self, name, source_opt, zr_flag=True, zr_range_flag=True, custom_r=None):
        pos = self.pop_Pos(zr_flag=zr_flag, zr_range_flag=zr_range_flag, custom_r=custom_r)
        write_fieldflp(name, source_opt, pos)
        return

    def run_model(self, model, dir_name, name, beam=None,zr_flag=True,zr_range_flag=True, custom_r=None, tilt_angle=0):
        """
        Inputs
        model - string
        dir_name - string
            Folder in which to store the .env, .flp, and other at_files for the model runs
            Should be an absolute path, and must end with a '/' for concatenation with tmp_fname
        name - string 
            Name of the .env file that will be generated by env.write_env_file
        beam - Beam object (pyat)
            beam attributes for bellhop run (angle, ds, etc. see the pyat source)
        zr_flag - Boolean
            Should I evaluate the field at only the receiver depths (True) or also at all the intermediate depths required for computing the model (False)?
        zr_range_flag - Boolean
            same but for range
        custom_r - numpy array
            range in METERS
        Output:
        x - np array of complex128
            Field evaluated at receiver depths and ranges specified in modification of env object.
        pos - Pos object from pyat env
        Instantiates a new attribute self.sim_x to hold computed field and self.sim_pos for pos object
        """             
        if model=='kraken':
            self.write_env_file(dir_name+name, model=model, zr_flag=zr_flag, zr_range_flag=zr_range_flag, custom_r=custom_r)
            system('cd ' + dir_name + ' && krakenc.exe ' + name)
            self.write_flp(dir_name + name, 'R',zr_flag=zr_flag,zr_range_flag=zr_range_flag, custom_r=custom_r)
            system('cd ' + dir_name + ' && field.exe ' + name)
            [ PlotTitle, PlotType, freqVec, atten, pos, pressure ] = read_shd(dir_name + name + '.shd')
            x = np.squeeze(pressure)
        elif model=='pe':
            print('hey man you should implement this')
        elif model=='bellhop':
            self.write_env_file(dir_name+name, model=model, zr_flag=zr_flag, beam=beam, zr_range_flag=zr_range_flag,custom_r=custom_r)
            system('cd ' + dir_name + ' && bellhop.exe ' + name)
            x = None
            pos = None
        elif model=='kraken_custom_r':
            self.write_env_file(dir_name+name, model='kraken', zr_flag=zr_flag, zr_range_flag=zr_range_flag, custom_r=custom_r)
            system('cd ' + dir_name + ' && krakenc.exe ' + name)
            fname = dir_name + name + '.mod'
            modes = read_modes(**{'fname':fname, 'freq':self.freq})
            self.modes = modes
            pos = self.pos
            if zr_flag == True:
                x = get_custom_r_modal_field(modes, custom_r, self.zs, self.zr, tilt_angle=tilt_angle)
            else:
                x = get_custom_r_modal_field(modes, custom_r, self.zs, pos.r.depth, tilt_angle=tilt_angle)
        else:
            raise ValueError('model input isn\'t supported')
        return x, pos
                
    def add_iw_field(self, fname):
        """
        Input
        fname : string
            path of feather file containing IW output
        """
        df = feather.read_dataframe(fname)
        z_ss = df['z'].unique()
        rp_ss = df['x'].unique()
        print(df['y'].unique().size)
        print('----', z_ss.size, rp_ss.size, df['c'].unique().size, df['t'].size, df.keys())
        ssp = df['c'].unique().reshape(rp_ss.size, z_ss.size)
        self.z_ss = z_ss
        self.rp_ss = rp_ss
        print(rp_ss)
        self.cw = ssp
        return

    def gen_env_fig(self, rs):
        """
        Generate a figure showing the SSP, receiver locations, source location
         and some text with the source frequency
        Pass in the source range zr """
        fig, ax1 = plt.subplots()
        ax2 = ax1.twiny()
        zr = self.zr
        zs = self.zs
        x, y = get_sea_surface(self.cw)
        ax1.plot([np.min(self.cw), np.max(self.cw)], [self.z_sb[0]]*2, color='tab:brown')
        ax1.plot(x, y, color='b')
        ax1.set_xlabel('SSP (m/s)')
        ax1.set_ylabel('Depth (m)')
        ax2.set_xlabel('Range (m)')
        ax1.invert_yaxis()
        ax2.scatter(0, zr[0], color='k', alpha=1, label='SSP')
        ax2.scatter([0]*zr.size, zr, color='r', label='Receive array')
        ax2.scatter(rs, zs, color='b', marker='+', label='Source position')
        ax1.plot(self.cw, self.z_ss, color='k', label='SSP')
        fig.suptitle('Environmental configuration\n(Source frequency is ' + str(self.freq) + ' Hz)')
        ax2.legend()
        return fig

        
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

    def change_depth(self, D):
        """ Update the Swellex environment 
        to be at a new depth D
        Shift bottom down accordingly"""
        self.z_ss = np.array([x for x in self.z_ss if x <= D])
        diff = D-self.z_sb[0]
        self.z_sb += diff
        self.cw = self.cw[:self.z_ss.size]
        """ Make sure cw has a point at D """
        """ AT rounds to nearest hundredth """
        if round(float(D),2) not in [round(x,2) for x in list(self.z_ss)]:
            self.z_ss = np.append(self.z_ss, D)
            self.cw = np.append(self.cw, self.cw[-1]).reshape(self.cw.size+1, 1)
        self.zmax = D
        self.add_field_params(self.dz, D, self.dr, self.rmax)
        #self.z_sb = z_sb
        #self.rp_sb = rp_sb
        #self.cb = cb
        #self.rhob = rhob
        #self.attn = attn
        #self.rbzb = rbzb
        self.env_dict = self.init_dict()
        return

    def s5_approach_adiabatic_replicas(self, dir_name, name, r, tilt_angle=0): 
        """
        For source candidates at elements of array pos.r.depth, 
        and given array r
        compute the field received at array zr
        Use the s5 bathymetry data to get source depth at each
        candidate range
        Assume linear bathymetry from D(receiver) to D source
        Omit modes that aren't excited at the source location

        I assume that the receiver positions have set as the source positions zs
        This ensures that the modes have been evaluated at the receiver depths
        Setting zr_flag to false below ensures that they have also been evaluated at each 
        candidate source depth

        """

        """ Get modes at receive array """
        self.run_model('kraken_custom_r', dir_name, name, zr_range_flag=False, zr_flag = False, custom_r = np.array([100]))
        rcvr_modes = self.modes
        num_rcvr_modes = rcvr_modes.M
        rcvr_kr = rcvr_modes.k
        phi_rcvr = rcvr_modes.get_receiver_modes(self.zs)
        print('num_rcvr_modes', num_rcvr_modes)

        dofr = s5_approach_D()
        depths = dofr(r)
        min_depth = np.min(depths)
        self.add_field_params(self.dz, min_depth, self.dr, self.rmax)
        self.pop_Pos(zr_flag=False)
        official_depth_grid = self.pos.r.depth # depth changes
        num_depths = self.pos.r.depth.size
        last_D = -10 # dummy value

        p = np.zeros((len(self.zs), self.pos.r.depth.size, r.size), dtype=np.complex128)

        """ Tilt angle correction """
        range_correction = get_range_correction(self.zs, tilt_angle)


        for r_ind, source_r in enumerate(r):
            D = dofr(source_r) 

            if abs(D - last_D) > 1:
                """ Recomputing modes """
                last_D = D
                self.change_depth(D)
                """ OVerride change to field params """
                self.add_field_params(self.dz, min_depth, self.dr, self.rmax)
                self.pop_Pos(zr_flag=False)
                self.run_model('kraken_custom_r', dir_name, name, zr_range_flag=False, zr_flag = False, custom_r = np.array([100]))
                source_modes = self.modes
                num_source_modes = source_modes.M
                phi_range = source_modes.get_receiver_modes(self.pos.r.depth[:num_depths])
                source_krs = source_modes.k
                num_modes = np.min([num_source_modes, num_rcvr_modes])
                source_krs = source_modes.k[:num_modes]
                relevant_rcvr_kr = rcvr_kr[:num_modes] 
                delta_kr = source_krs - relevant_rcvr_kr
                delta_kr = delta_kr.real # no need to worry about attenuation correction...I hope
                delta_kr = delta_kr.reshape((1, delta_kr.size))
                """ Get mode source excitation for each candidate depth """
                phi_source = source_modes.get_receiver_modes(self.pos.r.depth) 
                phi_source = phi_source[:, :num_modes]

            for rcvr_ind, source_depth in enumerate(self.zs):
                """ Get mode strength at the receiver specific """
                phi_tmp = rcvr_modes.get_source_strength(source_depth)
                """ Truncate shapes """
                phi_tmp = phi_tmp[0,:num_modes] 
                modal_matrix = phi_tmp * phi_source #column mat
                corrected_kr = relevant_rcvr_kr + delta_kr/2
                corrected_source_r = source_r - range_correction[rcvr_ind]
                r_mat= np.outer(corrected_kr, corrected_source_r)
                #depth_correction_mat = np.outer(delta_kr /2, source_r)
                """
                Note...kraken c has the attenuation as a negative
                imaginary part to k
                The implied convention is exp(-i k r) transform...
                """
                range_dep = np.exp(complex(0,1)*r_mat.conj()) / np.sqrt(r_mat.real)
                source_p = modal_matrix@range_dep
                source_p *= -np.exp(complex(0, 1)*np.pi/4)
                source_p /= np.sqrt(8*np.pi)
                source_p = source_p.conj()
                p[rcvr_ind, :, r_ind] = source_p[:,0]
        return p, self.pos
 
def env_from_json(name):
    env_dict = read_json(name)
    for key in env_dict.keys():
        print(type(env_dict[key]))
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
        

        
    
