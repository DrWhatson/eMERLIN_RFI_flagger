import numpy as np
import pylab as plt
import smooth
import pyfits as fits
import sys
import jdcal

ant_ID = {'LO':1, 'MK2':2, 'KN':5, 'DE':6, 'PI':7, 'DA':8, 'CM':9}
ant_no = {1:1, 2:2, 5:3, 6:4, 7:5, 8:6, 9:7}
ant_name = ['LO','MK2','KN','DE','PI','DA','CM']

class VisData:
    """Class to store visiility data for RFI module"""
    def __init__(self,hdu):
        self.hdu = hdu

    def get_nvis(self):
        nobs = len(self.hdu)-5
        nvis = 0
        for i in np.arange(nobs):
            nvis += self.hdu[i+5].header['NAXIS2']
        
        self.nobs = nobs
        self.nvis_tot = nvis

    def get_start_time(self):
        self.start_time = self.hdu[5].data.TIME[0]
        self.obs_date_str = self.hdu[5].data.field('DATE-OBS')
        self.obs_RA = self.hdu[5].data.CRVAL5
        self.obs_DEC = self.hdu[5].data.CRVAL6

        yr,mn,dy = self.obs_date_str.split('-')
        self.obs_JD = sum(jdcal.gcal2jd(yr,mn,dy))

    def get_freq(self):
        self.freq = self.hdu[5].header['REF_FREQ']
        self.dfrq = self.hdu[5].header['CHAN_BW']
        self.pfrq = self.hdu[5].header['REF_PIXL']

        self.nstoke = self.hdu[5].header['MAXIS2']
        self.nchan  = self.hdu[5].header['MAXIS3']
        self.nif    = self.hdu[5].header['MAXIS4']

    def get_ants(self):
        self.nant = len(self.hdu[1].data)-1
        
        
    def reset_flags(self):
        for i in range(len(self.flg)):
            self.flg[i][:,:] = 1

    def write_flag_table(self):
        """Append flags into new FG fits file"""

        imdata = np.zeros((1,1,1,8,512,4,3))
        pnames = ['UU---SIN','VV---SIN','WW---SIN',
                  'DATE','DATE','BASELINE','SOURCE',
                  'FREQSEL','INTTIM','CORR-ID']
        pdata = [0.,0.,0.,self.JD_obs,0.,0.,0.,0.,0.,0.]
        
        gdata = fits.GroupData(imdata,parnames=pnames,
                               pardata=pdata,bitpix=-32)
        
        prihdu = fits.GroupHDU(gdata)

        prihdu.set('BLOCKED',value=True,comment='Tape may be blocked')
        prihdu.set('OBJECT',value='MULTI',comment='Source name')
        prihdu.set('TELESCOP',value='e-MERLIN',comment=' ')
        prihdu.set('INSTRUME',value='VLBA',comment=' ')
        prihdu.set('OBSERVER',value='Calibrat',comment=' ')
        prihdu.set('DATE-OBS',value=self.obs_date_str,
                   comment='Obs start date YYYY-MM-DD')

        prihdu.set('BSCALE',value=1.0E0,
                   comment='REAL = TAPE * BSCALE + BZERO')
        prihdu.set('BZERO',value=0.0E0,comment=' ')
        prihdu.set('BUNIT',value='UNCALIB',comment='Units of flux')

        prihdu.set('EQUINOX',value=2.0E3,comment='Epoch of RA DEC')
        prihdu.set('ALTRPIX'value=1.0E+0,
                   comment='Altenate FREQ/VEL ref pixel')
        
        prihdu.set('OBSRA',value=self.obs_RA,
                    comment='Antenna pointing RA')
        prihdu.set('OBSDEC',value=self.obs_DEC,
                    comment='Antenna pointing DEC')
    
        prihdu.set('CRVAL2',value=1.0E+0,comment=' ') 
        prihdu.set('CDELT2',value=1.0E+0,comment=' ') 
        prihdu.set('CRPIX2',value=1.0E+0,comment=' ') 
        prihdu.set('CROTA2',value=0.0E+0,comment=' ') 

        prihdu.set('CRVAL3',value=-1.0E+0,comment=' ') 
        prihdu.set('CDELT3',value=-1.0E+0,comment=' ') 
        prihdu.set('CRPIX3',value=1.0E+0,comment=' ') 
        prihdu.set('CROTA3',value=0.0E+0,comment=' ') 

        prihdu.set('CRVAL4',value=self.freq,comment=' ') 
        prihdu.set('CDELT4',value=self.dfrq,comment=' ') 
        prihdu.set('CRPIX4',value=self.pfrq,comment=' ') 
        prihdu.set('CROTA4',value=0.0E+0,comment=' ') 

        prihdu.set('CRVAL5',value=1.0E+0,comment=' ') 
        prihdu.set('CDELT5',value=1.0E+0,comment=' ') 
        prihdu.set('CRPIX5',value=1.0E+0,comment=' ') 
        prihdu.set('CROTA5',value=0.0E+0,comment=' ') 

        prihdu.set('CRVAL6',value=self.obs_RA,comment=' ') 
        prihdu.set('CDELT6',value=1.0E+0,comment=' ') 
        prihdu.set('CRPIX6',value=1.0E+0,comment=' ') 
        prihdu.set('CROTA6',value=0.0E+0,comment=' ') 

        prihdu.set('CRVAL7',value=self.obs_DEC,comment=' ') 
        prihdu.set('CDELT7',value=1.0E+0,comment=' ') 
        prihdu.set('CRPIX7',value=1.0E+0,comment=' ') 
        prihdu.set('CROTA7',value=0.0E+0,comment=' ') 

        prihdu.set('PSCAL1',value=1./self.freq,comment=' ') 
        prihdu.set('PZERO1',value=0.0E+0,comment=' ') 

        prihdu.set('PSCAL2',value=1./self.freq,comment=' ') 
        prihdu.set('PZERO2',value=0.0E+0,comment=' ') 

        prihdu.set('PSCAL2',value=1./self.freq,comment=' ') 
        prihdu.set('PZERO2',value=0.0E+0,comment=' ') 

        prihdu.set('PSCAL3',value=1.0E+0,comment=' ') 
        prihdu.set('PZERO3',value=self.JD_obs,comment=' ') 

        prihdu.set('PSCAL4',value=1.0E+0,comment=' ') 
        prihdu.set('PZERO4',value=0.0E+0,comment=' ') 

        prihdu.set('PSCAL5',value=1.0E+0,comment=' ') 
        prihdu.set('PZERO5',value=0.0E+0,comment=' ') 

        prihdu.set('PSCAL6',value=1.0E+0,comment=' ') 
        prihdu.set('PZERO6',value=0.0E+0,comment=' ') 

        prihdu.set('PSCAL7',value=1.0E+0,comment=' ') 
        prihdu.set('PZERO7',value=0.0E+0,comment=' ') 

        prihdu.set('PSCAL8',value=1.0E+0,comment=' ') 
        prihdu.set('PZERO8',value=0.0E+0,comment=' ') 

        prihdu.set('PSCAL9',value=1.0E+0,comment=' ') 
        prihdu.set('PZERO9',value=0.0E+0,comment=' ') 

        prihdu.set('PSCAL10',value=1.0E+0,comment=' ') 
        prihdu.set('PZERO10',value=0.0E+0,comment=' ') 

        for i in range(len(self.bl)):
          a1,a2 = self.base_name[self.bl[i]]
          ant1 = ant_ID[a1]
          ant2 = ant_ID[a2]
          print i, a1,a2,ant1,ant2
          if self.amp[i].shape[0]==0:
            continue

          ix,iy = np.where(np.transpose(self.flg[i])==0)  
          ixu = np.unique(ix)  

          for j in ixu:
            col = np.where(ix==j)[0]
            blocks = iy[col]-range(len(col))
            indx = np.unique(blocks)
            prek = 0
            for k in indx:
                dk = len(np.where(blocks==k)[0])
                ik = k+prek
                src.append(0)
                subary.append(1)
                frqid.append(1)
                ifs.append([j/128+1,j/128+1])
                chan = (j*4)%512
                chans.append([chan+1,chan+5])
                pflags.append([1,1,1,1])
                reasons.append('RFI')
                ants.append([ant1,ant2])
                t1 = self.start_time+self.vtim[i][ik]*self.dt
                t2 = self.start_time+(self.vtim[i][ik+dk-1]+1)*self.dt
                timrng.append([t1,t2])
                prek += dk


        col1 = fits.Column(name='SOURCE',format='1J',unit=' ',
                           array=src)
        col2 = fits.Column(name='SUBARRAY',format='1J',unit=' ',
                           array=subary)
        col3 = fits.Column(name='FREQ ID',format='1J',unit=' ',
                           array=frqid)
        col4 = fits.Column(name='ANTS',format='2J',unit=' ',
                           array=ants)
        col5 = fits.Column(name='TIME RANGE',format='2E',
                           unit='DAYS',array=timrng)
        col6 = fits.Column(name='IFS',format='2J',unit=' ',
                           array=ifs)
        col7 = fits.Column(name='CHANS',format='2J',unit=' ',
                           array=chans)
        col8 = fits.Column(name='PFLAGS',format='4X',unit=' ',
                           array=pflags)
        col9 = fits.Column(name='REASON',format='24A',unit=' ',
                           array=reasons)

        cols = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9])
        
        fg_hdu = fits.new_table(cols) 
        fg_hdu.header.set('EXTNAME',value='AIPS FG',
                          comment='AIPS table file')
        fg_hdu.header.set('EXTVER',value=1,
                          comment='Version number of table')

        hdulist = fits.HDUList([prihud,fg_hdu])
        hdulist.writeto('Test_Flags.fits')

def read_fits(fits_file,dch=4,dt=10,progress=False,MAD=0):
#    """Routine to read given uvdata into a VisData class"""

    hdu = fits.open(fits_file)

  # Extract header information to put in visdata  

    vd = VisData(hdu)
    vd.src = fits_file
    vd.dchan = dch
    vd.dt = dt/86400.
    vd.get_nvis()
    vd.get_start_time()
    vd.get_freq()
    vd.get_ants()
    vd.nbas = vd.nant*(vd.nant-1)/2 + vd.nant
    vd.nvis = vd.nvis_tot/vd.nbas
#    vd.nif,vd.nchan,vd.nstoke,vd.nval = uvdata[0].visibility.shape

    vd.base_ant = []
    vd.base_name = []
    for i in np.arange(vd.nant+1):
        for j in np.arange(i+1,vd.nant+1):
            vd.base_ant.append((i+1,j+1))
            print "b/l=",ant_name[i],ant_name[j]
            vd.base_name.append((ant_name[i],ant_name[j]))

  # Intialise lists to contain visibility data

    vis  = [[] for i in range(vd.nbas)]
    vis2 = [[] for i in range(vd.nbas)]
    vhit = [[] for i in range(vd.nbas)]
    vd.vtim = [[] for i in range(vd.nbas)]

    nb = 0
    tdic = [{} for i in range(vd.nbas)]

  # Read in visibility arrays and append lists

    ipos = np.zeros(vd.nbas,'i')
    itot = 0

    for io in np.arange(vd.nobs):
      nvis_obs = vd.hdu[5+io].header['NAXIS2']  
      for i in np.arange(nvis_obs):
        itot += 1  
        baseline = vd.hdu[5+io].data.BASELINE[i]  
        ant1 = baseline/256
        ant2 = baseline%256
        

        if ant1==ant2:
            continue
        
        if ant1<ant2:
            a1 = ant_no[ant1]
            a2 = ant_no[ant2]
        else:
            a1 = ant_no[ant2]
            a2 = ant_no[ant1]

        ib = (2*vd.nant+2-a1)*(a1-1)/2 + a2 - a1 -1

#        print i, a1, a2, ib

        t = vd.hdu[io+5].data.TIME[i]
        it = int((t-vd.start_time)/vd.dt)
        flux = vd.hdu[io+5].data.FLUX[i,:]
        flux.shape = (8,512,4,2)
        a = np.sqrt(flux[:,:,:,0]**2+flux[:,:,:,1]**2)
        a = a.reshape(vd.nif*vd.nchan/4,4,vd.nstoke).sum(axis=1)

        if progress:
          if itot%1000==0:
            pc = itot*101/vd.nvis_tot
            pc5 = pc/5
            sys.stdout.write('\r')
            sys.stdout.write('Loading %s [%-20s] %d%%' % (vd.src,'='*pc5,pc))
            sys.stdout.flush()
                           
        if not tdic[ib].has_key(it):
            tdic[ib][it] = ipos[ib]
            vis[ib].append(a)
            vis2[ib].append(a**2)
            vhit[ib].append(4)
            vd.vtim[ib].append(it)
            ipos[ib] += 1
        else:
            ip = tdic[ib][it]
            vis[ib][ip] += a
            vis2[ib][ip] += a**2
            vhit[ib][ip] += 4

    vd.end_time = vd.hdu[vd.nobs+4].data.TIME[-1]
    vd.nt = int((vd.end_time-vd.start_time)/vd.dt)+1

    print '\n Calculating stats'

  # Convert lists to arrays

    for i in np.arange(vd.nbas):
        vis[i] =  np.array(vis[i])
        vis2[i] = np.array(vis2[i])
        vhit[i] = np.array(vhit[i])
        vhit[i] = vhit[i][:,np.newaxis,np.newaxis]
        vd.vtim[i] = np.array(vd.vtim[i])

  # For each baseline generate amp and error arrays in visdata

    vd.amp = []
    vd.err = []
    vd.bl = []
    vd.flg = []

    print len(vd.base_ant)

    for i in np.arange(vd.nbas):

        print vd.base_ant[i][0],vd.base_ant[i][1]
        if vd.base_ant[i][0]==vd.base_ant[i][1]:
            continue
        
        vd.bl.append(i)

        Avis = vis[i]/vhit[i]

        Avis_err = np.sqrt(vis2[i]-vis[i]**2/vhit[i])    
        Avis_err /= np.sqrt(vhit[i]*(vhit[i]-1))

        Avis = np.where(vhit[i]==0,0,Avis)
        Avis_err = np.where(vhit[i]==0,1e10,Avis_err)
        Avis_err = np.where(Avis_err<1e-6,1e10,Avis_err)

        Avis_err = np.where(Avis==np.nan,1e10,Avis_err)
        Avis = np.where(Avis==np.nan,0,Avis)

        print i,Avis_err.shape
        vd.amp.append(Avis)
        vd.err.append(Avis_err)

        vd.flg.append(np.ones(Avis.shape[:2],dtype='b'))


    return vd


def flag_via_amp(vd,thres=1.5,sig=4,ndx=20):
    """Flag based on ratio of amps to smoothed weighted amps"""

    x = np.arange(-ndx,ndx+1)
    y = x[:,np.newaxis]
    r2 = x**2 + y**2
    g = np.exp(-r2/2/sig**2)

    for i in range(len(vd.amp)):
      for j in range(2):
        if vd.amp[i].shape[0]==0:
            continue
        a = vd.amp[i][:,:,j]
        e = vd.err[i][:,:,j]
        w   = 1./e**2
        smo = smooth.weighted(a,w,g)
        vd.flg[i] *= np.where(e/smo>thres,0,1)


def flag_via_rms(vd,thres=1.5,sig=4,ndx=20):
    """Flag based on ratio of errs to smoothed weighted"""

    x = np.arange(-ndx,ndx+1)
    y = x[:,np.newaxis]
    r2 = x**2 + y**2
    g = np.exp(-r2/2/sig**2)

    for i in range(len(vd.amp)):
      for j in range(2):  
        if vd.amp[i].shape[0]==0:
            continue
        e = vd.err[i][:,:,j]
        w   = 1./e**2
        smo = smooth.weighted(e,w,g)
        vd.flg[i] *= np.where(e/smo>thres,0,1)


def flag_via_s2n(vd,thres=0.8,sig=2,ndx=20):
    """Flag based on ratio of amps to smoothed weighted amps"""

    x = np.arange(-ndx,ndx+1)
    y = x[:,np.newaxis]
    r2 = x**2 + y**2
    g = np.exp(-r2/2/sig**2)

    for i in range(len(vd.amp)):
      for j in range(2):  
        if vd.amp[i].shape[0]==0:
            continue
        s2n = vd.amp[i][:,:,j]/vd.err[i][:,:,j]
        e = vd.err[i][:,:,j]
        w   = 1./e**2
        smo = smooth.weighted(s2n,w,g)
        vd.flg[i] *= np.where(s2n/smo<thres,0,1)

def flag_via_amp_median_threshold(vd,thres=2.0):
    """Flag above signal*thres*median(signal)"""
    for i in range(len(vd.amp)):
      ix,iy = np.where(vd.flg[i][:,:]==1)
      for j in range(2):
        if vd.amp[i].shape[0]==0:
            continue
        med_amp = np.median(vd.amp[i][ix,iy,j])  
        vd.flg[i] *= np.where((vd.amp[i][:,:,j]>med_amp*thres),0,1)

def flag_via_rms_median_threshold(vd,thres=2.0):
    """Flag above err*thres*median(err)"""
    for i in range(len(vd.err)):
      ix,iy = np.where(vd.flg[i][:,:]==1)  
      for j in range(2):
        if vd.amp[i].shape[0]==0:
            continue
        med_err = np.median(vd.err[i][ix,iy,j])  
        vd.flg[i] *= np.where((vd.err[i][:,:,j]>med_err*thres),0,1)


