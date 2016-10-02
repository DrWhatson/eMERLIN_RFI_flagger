import numpy as np
import numpy.ma as ma
from scipy.interpolate import UnivariateSpline
import wx
import wx.lib.buttons as buttons
import pyfits as fits

import matplotlib
matplotlib.interactive(False)
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
from matplotlib.pyplot import gcf,setp

import RFI_fits as RFI
#import RFI
#import smooth

import cPickle as pic
import sys

global uv,baseline,bp_window


class Baseline_Panel(wx.Panel):
    def __init__(self, *args, **kwargs):
        wx.Panel.__init__(self, *args, **kwargs)
        self.parent = args[0]

        self.rfi_Window = self.parent.rfi_Window
        a1,a2 = uv.base_name[uv.bl[self.rfi_Window.baseline]]
        base_name = "%i %s-%s" % (bl,a1,a2)

        self.heading = wx.StaticText(self,label='Baseline',pos=(20,0))

        self.next_button = wx.Button(self,-1,'Prev',pos=(0,20),size=(40,24))
        self.prev_button = wx.Button(self,-1,'Next',pos=(50,20),size=(40,24))
        self.baseText = wx.StaticText(self,-1,base_name,pos=(10,50),size=(60,24))
        self.rb1 = wx.RadioButton(self, -1, 'RR', (10, 70), style=wx.RB_GROUP)
        self.rb2 = wx.RadioButton(self, -1, 'LL', (50, 70))

        self.next_button.Bind(wx.EVT_BUTTON,self.set_baseline)
        self.prev_button.Bind(wx.EVT_BUTTON,self.set_baseline)

        self.rb1.Bind(wx.EVT_RADIOBUTTON,self.set_pol)
        self.rb2.Bind(wx.EVT_RADIOBUTTON,self.set_pol)

    def set_baseline(self,e):

        btn = e.GetEventObject()
        lab = btn.GetLabel()

        nbas = len(uv.amp)
        bl = self.rfi_Window.baseline

        if lab=="Next":
            bl += 1
        elif lab=="Prev":
            bl -= 1

        if bl==nbas:
            bl = 0

        if bl<0:
            bl = nbas-1

        rfi_window = self.parent.rfi_Window
        tpanel = self.parent.controls.threshold_panel

        if rfi_window.amp_thres[bl]==0:
            rfi_window.amp_thres[bl] = float(tpanel.amp_thres_entry.GetValue())
        else:
            tpanel.set_amp_thres(rfi_window.amp_thres[bl])

        if rfi_window.rms_thres[bl]==0:
            rfi_window.rms_thres[bl] = float(tpanel.rms_thres_entry.GetValue())
        else:
            tpanel.set_rms_thres(rfi_window.rms_thres[bl])

        if rfi_window.dropout_thres[bl]==0:
            rfi_window.dropout_thres[bl] = float(tpanel.dropout_thres_entry.GetValue())
        else:
            tpanel.set_dropout_thres(rfi_window.dropout_thres[bl])
        
        a1,a2 = uv.base_name[uv.bl[bl]]
        base_name = "%i %s-%s" % (bl,a1,a2)
        self.baseText.SetLabel(base_name)
        self.rfi_Window.set_baseline(bl)
        bp_window.set_baseline(bl)

    def set_pol(self,evt):
        btn = evt.GetEventObject()
        lab = btn.GetLabel()

        if lab=='RR':
            pol = 0
        elif lab=='LL':
            pol=1

        self.rfi_Window.set_polarization(pol)

        
class Threshold_Panel(wx.Panel):
    def __init__(self, *args, **kwargs):
        wx.Panel.__init__(self, *args, **kwargs)
        self.parent = args[0]
        self.rfi_Window = self.parent.rfi_Window

        self.heading = wx.StaticText(self,label='Threshold levels',pos=(5,0))
 
        self.amp_thres_text = wx.StaticText(self,label="Amp threshold", pos=(5,30))
        self.amp_thres_entry = wx.TextCtrl(self,value="2.20",pos=(110,30),style=wx.TE_PROCESS_ENTER)
        self.amp_thres_entry.Bind(wx.EVT_TEXT_ENTER,self.set_amp_thres)

        self.rms_thres_text = wx.StaticText(self,label="RMS threshold", pos=(5,60))
        self.rms_thres_entry = wx.TextCtrl(self,value="15.00",pos=(110,60),style=wx.TE_PROCESS_ENTER)
        self.rms_thres_entry.Bind(wx.EVT_TEXT_ENTER,self.set_rms_thres)

        self.dropout_thres_text = wx.StaticText(self,label="Dropout thres", pos=(5,90))
        self.dropout_thres_entry = wx.TextCtrl(self,value="3.00",pos=(110,90),style=wx.TE_PROCESS_ENTER)
        self.dropout_thres_entry.Bind(wx.EVT_TEXT_ENTER,self.set_dropout_thres)

        self.IF_thres_text = wx.StaticText(self,label='IF thresholds',pos=(5,130))
        self.IF_thres_button = wx.Button(self,-1,'Apply',pos=(110,125),size=(60,30))
        self.IF_thres_button.Bind(wx.EVT_BUTTON,self.apply_IF_thres)

        
    def set_amp_thres(self,evt):
        val = float(self.amp_thres_entry.GetValue())
        self.rfi_Window.get_amp_threshold(val)

    def set_rms_thres(self,evt):
        val = float(self.rms_thres_entry.GetValue())
        self.rfi_Window.get_rms_threshold(val)

    def set_dropout_thres(self,evt):
        val = float(self.dropout_thres_entry.GetValue())
        self.rfi_Window.get_dropout_threshold(val)

    def apply_IF_thres(self,evt):
        self.rfi_Window.apply_IF_thres(bp_window.IF_thres)


class File_Panel(wx.Panel):
    def __init__(self, *args, **kwargs):
        wx.Panel.__init__(self, *args, **kwargs)
        self.parent = args[0]

        self.heading = wx.StaticText(self,label='Settings',pos=(50,0))
 
        self.load_button = wx.Button(self,-1,'Load',pos=(0,20),size=(40,24))
        self.load_button.Bind(wx.EVT_BUTTON,self.load_settings)
        
        self.save_button = wx.Button(self,-1,'Save',pos=(100,20),size=(40,24))
        self.save_button.Bind(wx.EVT_BUTTON,self.save_settings)
        
        self.filename = wx.StaticText(self,label='Default',pos=(0,60))


    def save_settings(self,evt):
        
        saveFileDialog = wx.FileDialog(self, "Save settings pic file", "", "",
                        "Settings (*.pic)|*.pic", wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)

        if saveFileDialog.ShowModal() == wx.ID_CANCEL:
            return

        filename = saveFileDialog.GetPath()
        pf = open(filename,'wb')

        amp_thres = self.parent.rfi_Window.amp_thres
        rms_thres = self.parent.rfi_Window.rms_thres
        dropout_thres = self.parent.rfi_Window.dropout_thres

        pic.dump(amp_thres,pf)
        pic.dump(rms_thres,pf)
        pic.dump(dropout_thres,pf)
        
        pf.close()
        
        filename = filename.split('/')[-1]
        self.filename.SetLabel(filename)


    def load_settings(self,evt):
        
        openFileDialog = wx.FileDialog(self, "Open settings pic file", "", "",
                        "Settings (*.pic)|*.pic", wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        if openFileDialog.ShowModal() == wx.ID_CANCEL:
            return

        filename = openFileDialog.GetPath()
        pf = open(filename,'rb')

        amp_thres = pic.load(pf)
        rms_thres = pic.load(pf)
        dropout_thres = pic.load(pf)
        pf.close()

        bl = self.parent.rfi_Window.baseline

        self.parent.rfi_Window.amp_thres = amp_thres
        self.parent.controls.threshold_panel.amp_thres_entry.SetValue("%2.2f" % amp_thres[bl])
        self.parent.rfi_Window.rms_thres = rms_thres
        self.parent.controls.threshold_panel.rms_thres_entry.SetValue("%2.2f" % rms_thres[bl])
        self.parent.rfi_Window.amp_thres = dropout_thres
        self.parent.controls.threshold_panel.dropout_thres_entry.SetValue("%2.2f" % dropout_thres[bl])

class Flag():
    def __init__(self):
        self.bchan = []
        self.echan = []
        self.btime = []
        self.etime = []



class Flags_Panel(wx.Panel):
    def __init__(self, *args, **kwargs):
        wx.Panel.__init__(self, *args, **kwargs)
        self.parent = args[0]

        self.heading = wx.StaticText(self,label='Write Flags',pos=(20,0))
        self.write_button = wx.Button(self,-1,'Go',pos=(0,30),size=(100,35))
        self.write_button.Bind(wx.EVT_BUTTON,self.write_new_flags)
        self.status = wx.StaticText(self,label='       ',pos=(10,70))

    def write_flags(self,evt):
        self.status.SetLabel('Writing flags')
        uv.write_flag_table()
        self.status.SetLabel('Done')

    def write_new_flags(self,evt):
        self.status.SetLabel('Re-reading data')
                
        ipos = np.zeros(uv.nbas,'i')
        itot = 0

        flags = []
        IF_clip = np.zeros((8,451,2,21),'f')
        for i in range(21):
            flags.append([])
            for j in range(8):
                flags[i].append(Flag())

                a = bp_window.IF_thres[j,0,i,:]
                amp_max = self.parent.rfi_Window.amp_thres_max[i,0]
                IF_clip[j,:,0,i] = np.polyval(a,np.arange(451))
                IF_clip[j,:,0,i] = np.where(IF_clip[j,:,0,i]>amp_max,amp_max,IF_clip[j,:,0,i])

                a = bp_window.IF_thres[j,1,i,:]
                IF_clip[j,:,1,i] = np.polyval(a,np.arange(451))
                IF_clip[j,:,1,i] = np.where(IF_clip[j,:,1,i]>amp_max,amp_max,IF_clip[j,:,1,i])
                


        for io in np.arange(uv.nobs):
            nvis_obs = uv.hdu[5+io].header['NAXIS2']  
            for i in np.arange(nvis_obs):
                itot += 1  
                baseline = uv.hdu[5+io].data.BASELINE[i]  
                ant1 = baseline/256
                ant2 = baseline%256

                if ant1==ant2:
                    continue
        
                if ant1<ant2:
                    a1 = RFI.ant_no[ant1]
                    a2 = RFI.ant_no[ant2]
                else:
                    a1 = RFI.ant_no[ant2]
                    a2 = RFI.ant_no[ant1]

                if ant1==1 and ant2==2:
                    continue

                ib = (2*uv.nant+2-a1)*(a1-1)/2 + a2 - a1 -1

                t = uv.hdu[io+5].data.TIME[i]+uv.hdu[io+5].data.DATE[i]
                t -= uv.start_date
                flux = uv.hdu[io+5].data.FLUX[i,:]
                flux.shape = (8,512,4,2)
                a = np.sqrt(flux[:,30:-31,:2,0]**2+flux[:,30:-31,:2,1]**2)

                # Get common flags
                flg  = np.where(a[:,:,0]>IF_clip[:,:,0,ib],0,1)
                flg *= np.where(a[:,:,1]>IF_clip[:,:,1,ib],0,1)

                # Loop over IFs looking for blocks of RFI
                for j in range(8):
                    ix = np.where(flg[j,:]==0)[0]
                    if len(ix)==0:
                        continue
                    blk = ix - range(len(ix)) 
                    indx,indx2 = np.unique(blk,return_index=1)  # Start channels of RFI
                    indx2[:-1] = indx2[1:]
                    indx2[-1] = len(ix)
                    echan = indx + indx2  
                    indx2[1:] = indx2[1:] - indx2[:-1]
                    bchan = echan - indx2

                    nchan = len(bchan)

                    for k in range(nchan):

                        flags[ib][j].bchan.append(bchan[k])
                        flags[ib][j].echan.append(echan[k]-1)

                        flags[ib][j].btime.append(t-0.25/86400.0)
                        flags[ib][j].etime.append(t+0.25/86400.0)
                    

        self.status.SetLabel('Writing flag table')

        self.write_flag_table(flags)

        self.status.SetLabel('Done')



    def write_flag_table(self,flags,flg_dir=''):
        """Append flags into new FG fits file"""

        imdata = np.zeros((1,1,1,8,512,4,3))
        pnames = ['UU---SIN','VV---SIN','WW---SIN',
                  'DATE','DATE','BASELINE','SOURCE',
                  'FREQSEL','INTTIM','CORR-ID']
        pdata = [0.,0.,0.,uv.obs_JD,0.,0.,0.,0.,0.,0.]
        
        gdata = fits.GroupData(imdata,parnames=pnames,
                               pardata=pdata,bitpix=-32)
        
        prihdu = fits.GroupsHDU(gdata)

        prihdu.header.set('BLOCKED',value=True,comment='Tape may be blocked')
        prihdu.header.set('OBJECT',value=uv.source,comment='Source name')
        prihdu.header.set('TELESCOP',value='e-MERLIN',comment=' ')
        prihdu.header.set('INSTRUME',value='VLBA',comment=' ')
        prihdu.header.set('OBSERVER',value='Calibrat',comment=' ')
        prihdu.header.set('DATE-OBS',value=uv.obs_date_str,
                   comment='Obs start date YYYY-MM-DD')

        prihdu.header.set('BSCALE',value=1.0E0,
                   comment='REAL = TAPE * BSCALE + BZERO')
        prihdu.header.set('BZERO',value=0.0E0,comment=' ')
        prihdu.header.set('BUNIT',value='UNCALIB',comment='Units of flux')

        prihdu.header.set('EQUINOX',value=2.0E3,comment='Epoch of RA DEC')
        prihdu.header.set('ALTRPIX',value=1.0E+0,
                   comment='Altenate FREQ/VEL ref pixel')
        
        prihdu.header.set('OBSRA',value=uv.obs_RA,
                    comment='Antenna pointing RA')
        prihdu.header.set('OBSDEC',value=uv.obs_DEC,
                    comment='Antenna pointing DEC')
    
        prihdu.header.set('CRVAL2',value=1.0E+0,comment=' ') 
        prihdu.header.set('CDELT2',value=1.0E+0,comment=' ') 
        prihdu.header.set('CRPIX2',value=1.0E+0,comment=' ') 
        prihdu.header.set('CROTA2',value=0.0E+0,comment=' ') 

        prihdu.header.set('CRVAL3',value=-1.0E+0,comment=' ') 
        prihdu.header.set('CDELT3',value=-1.0E+0,comment=' ') 
        prihdu.header.set('CRPIX3',value=1.0E+0,comment=' ') 
        prihdu.header.set('CROTA3',value=0.0E+0,comment=' ') 

        prihdu.header.set('CRVAL4',value=uv.freq,comment=' ') 
        prihdu.header.set('CDELT4',value=uv.dfrq,comment=' ') 
        prihdu.header.set('CRPIX4',value=uv.pfrq,comment=' ') 
        prihdu.header.set('CROTA4',value=0.0E+0,comment=' ') 

        prihdu.header.set('CRVAL5',value=1.0E+0,comment=' ') 
        prihdu.header.set('CDELT5',value=1.0E+0,comment=' ') 
        prihdu.header.set('CRPIX5',value=1.0E+0,comment=' ') 
        prihdu.header.set('CROTA5',value=0.0E+0,comment=' ') 

        prihdu.header.set('CRVAL6',value=uv.obs_RA,comment=' ') 
        prihdu.header.set('CDELT6',value=1.0E+0,comment=' ') 
        prihdu.header.set('CRPIX6',value=1.0E+0,comment=' ') 
        prihdu.header.set('CROTA6',value=0.0E+0,comment=' ') 

        prihdu.header.set('CRVAL7',value=uv.obs_DEC,comment=' ') 
        prihdu.header.set('CDELT7',value=1.0E+0,comment=' ') 
        prihdu.header.set('CRPIX7',value=1.0E+0,comment=' ') 
        prihdu.header.set('CROTA7',value=0.0E+0,comment=' ') 

        prihdu.header.set('PSCAL1',value=1./uv.freq,comment=' ') 
        prihdu.header.set('PZERO1',value=0.0E+0,comment=' ') 

        prihdu.header.set('PSCAL2',value=1./uv.freq,comment=' ') 
        prihdu.header.set('PZERO2',value=0.0E+0,comment=' ') 

        prihdu.header.set('PSCAL3',value=1./uv.freq,comment=' ') 
        prihdu.header.set('PZERO2',value=0.0E+0,comment=' ') 

        prihdu.header.set('PSCAL4',value=1.0E+0,comment=' ') 
        prihdu.header.set('PZERO3',value=uv.obs_JD,comment=' ') 

        prihdu.header.set('PSCAL5',value=1.0E+0,comment=' ') 
        prihdu.header.set('PZERO4',value=0.0E+0,comment=' ') 

        prihdu.header.set('PSCAL6',value=1.0E+0,comment=' ') 
        prihdu.header.set('PZERO6',value=0.0E+0,comment=' ') 

        prihdu.header.set('PSCAL7',value=1.0E+0,comment=' ') 
        prihdu.header.set('PZERO7',value=0.0E+0,comment=' ') 

        prihdu.header.set('PSCAL8',value=1.0E+0,comment=' ') 
        prihdu.header.set('PZERO8',value=0.0E+0,comment=' ') 

        prihdu.header.set('PSCAL9',value=1.0E+0,comment=' ') 
        prihdu.header.set('PZERO9',value=0.0E+0,comment=' ') 

        prihdu.header.set('PSCAL10',value=1.0E+0,comment=' ') 
        prihdu.header.set('PZERO10',value=0.0E+0,comment=' ') 

        # Initialise with flagging Lovell-Mk2

        src = [0]
        subary =[0]
        frqid = [-1]
        ifs = [[1,8]]
        chans = [[1,0]]
        pflags = [[1,1,1,1]]
        reasons = ['Lovell - Mk2 baseline']
        ants = [[1,2]]
        timrng = [[0.,9999.]]

        # Add IF bounaries +-30 chans

        for i in range(8):
            src.append(0)
            subary.append(0)
            frqid.append(-1)
            ifs.append([i,i])
            chans.append([1,30])
            pflags.append([1,1,1,1])
            ants.append([0,0])
            timrng.append([0.,9999.])
            reasons.append('Lower IF edge')

            src.append(0)
            subary.append(0)
            frqid.append(-1)
            ifs.append([i,i])
            chans.append([480,512])
            pflags.append([1,1,1,1])
            ants.append([0,0])
            timrng.append([0.,9999.])
            reasons.append('Upper IF edge')

        # Add dropouts if any

        for i in range(len(uv.bl)):
            a1,a2 = uv.base_name[uv.bl[i]]
            ant1 = RFI.ant_ID[a1]
            ant2 = RFI.ant_ID[a2]
            ix = np.where(uv.dflg[i]==0)[0]
            blocks = ix-np.arange(len(ix))
            indx = np.unique(blocks)

            prek = 0
            for k in indx:
                dk = len(np.where(blocks==k)[0])
                ik = k+prek
                src.append(0)
                subary.append(0)
                frqid.append(-1)
                ifs.append([1,8])
                chans.append([1,0])
                pflags.append([1,1,1,1])
                reasons.append('Dropout')
                ants.append([ant1,ant2])
                t1 = uv.start_time+uv.vtim[i][ik]*uv.dt
                t2 = uv.start_time+(uv.vtim[i][ik+dk-1]+1)*uv.dt
                timrng.append([t1,t2])
                uv.dflg[i][ik:ik+dk,0] = 1  # Clear so not repeated

                prek += dk


        for i in range(len(uv.bl)):
          a1,a2 = uv.base_name[uv.bl[i]]
          ant1 = RFI.ant_ID[a1]
          ant2 = RFI.ant_ID[a2]
          print i, a1,a2,ant1,ant2
          for j in range(8):
              nflag = len(flags[i][j].bchan)
              print nflag
              for k in range(nflag):
                src.append(0)
                subary.append(0)
                frqid.append(-1)
                ifs.append([j+1,j+1])
                bchan = flags[i][j].bchan[k]+31
                echan = flags[i][j].echan[k]+31
#                print bchan,echan
                chans.append([bchan,echan])
                pflags.append([1,1,1,1])
                reasons.append('RFI')
                ants.append([ant1,ant2])
                t1 = flags[i][j].btime[k]
                t2 = flags[i][j].etime[k]
                timrng.append([t1,t2])


        print len(src)

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

        hdulist = fits.HDUList([prihdu,fg_hdu])

        outfile = flg_dir+'Flags_%s.fits' % uv.source

        hdulist.writeto(outfile)



class Controls():
    def __init__(self,parent):
        self.parent = parent

        global bp_window

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer2 = wx.BoxSizer(wx.HORIZONTAL)

        self.baseline_panel = Baseline_Panel(parent)
        self.threshold_panel = Threshold_Panel(parent)
        self.file_panel = File_Panel(parent)
        self.flags_panel = Flags_Panel(parent)
        self.bp_window = bp_window

        sizer.Add(self.bp_window,1,wx.EXPAND)
        sizer.Add(wx.StaticLine(self.parent, -1, style=wx.LI_HORIZONTAL),1,wx.EXPAND)
        sizer.Add(self.threshold_panel,2,wx.EXPAND)
        sizer.Add(wx.StaticLine(self.parent, -1, style=wx.LI_HORIZONTAL),1,wx.EXPAND)

        sizer2.Add(self.baseline_panel,1,wx.EXPAND)
        sizer2.Add(wx.StaticLine(self.parent, -1, style=wx.LI_VERTICAL),1,wx.EXPAND)
        sizer2.Add(self.file_panel,2,wx.EXPAND)
        sizer2.Add(wx.StaticLine(self.parent, -1, style=wx.LI_VERTICAL),1,wx.EXPAND)
        sizer2.Add(self.flags_panel,2,wx.EXPAND)

        sizer.Add(sizer2,2,wx.EXPAND)
        sizer.Add(wx.StaticLine(self.parent, -1, style=wx.LI_HORIZONTAL),1,wx.EXPAND)

        self.sizer = sizer

class RFI_Frame(wx.Frame):

    def __init__(self, *args, **kwargs):
        wx.Frame.__init__(self, *args, **kwargs)
        
        global bp_window 
        
        bp_window = BPass_Window(self)
        self.rfi_Window = RFI_Window(self)
        self.controls = Controls(self)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.rfi_Window,0,wx.EXPAND)
        sizer.Add(self.controls.sizer,0,wx.EXPAND)

        self.SetSizer(sizer)


class RFI_Window(wx.Window):
    
    def __init__(self, *args, **kwargs):
        wx.Window.__init__(self, *args, **kwargs)

        self.baseline = 1

        nbas = len(uv.amp)        
        self.amp_thres = np.zeros(nbas,'f')
        self.amp_thres_max = np.zeros((nbas,2),'f')
        self.rms_thres = np.zeros(nbas,'f')
        self.dropout_thres = np.zeros(nbas,'f')
        self.pol = 0
        
        print uv.err[self.baseline].shape

        self.figure = Figure(figsize=(10,6),dpi=100)
        self.canvas = FigureCanvasWxAgg(self,-1,self.figure)

        main_box = wx.BoxSizer(wx.HORIZONTAL)
        main_box.Add(self.canvas, flag=wx.EXPAND, proportion=1)
        self.SetSizer(main_box)

        self.draw()


    def get_clean_amp(self, percnt_pnt=90.0, pol=0):
    # Clean bits of spectrum
        cb = [(1000,1300),(1400,1650),(3000,3150),(3250,3400)] 
        
        cb_index = []
        for i in np.arange(len(cb)):
            cb_index.append(np.arange(cb[i][0],cb[i][1]))

        bl = self.baseline

        cb_index = np.concatenate(cb_index)
        amp = np.median(uv.amp[bl][:,cb_index,pol],axis=1)

        thres = np.percentile(amp,percnt_pnt)

        return amp, thres

    def get_clean_rms(self, pol=0):
    # Clean bits of spectrum
        cb = [(1000,1300),(1400,1650),(3000,3150),(3250,3400)] 
        
        cb_index = []
        for i in np.arange(len(cb)):
            cb_index.append(np.arange(cb[i][0],cb[i][1]))

        bl = self.baseline

        cb_index = np.concatenate(cb_index)
        cln_rms = np.median(uv.err[bl][:,cb_index,pol])

        return cln_rms


    def flag_dropout(self, percnt_pnt=90.0, thres=0.3):

        bl = self.baseline
        thres = self.dropout_thres[bl]

        amp0, thres0 = self.get_clean_amp(pol=0)
        amp1, thres1 = self.get_clean_amp(pol=1)


        self.clean_amp = (thres0, thres1)

        uv.dflg[bl] = np.where((amp0[:,np.newaxis]<thres0/thres),0,1) 
        uv.dflg[bl][:,:] *= np.where((amp1[:,np.newaxis]<thres1/thres),0,1) 
        
        rms0 = self.get_clean_rms(pol=0)
        rms1 = self.get_clean_rms(pol=1)

        self.clean_rms = (rms0, rms1)

    def set_baseline(self,bl):
        self.baseline = bl
        self.flag_dropout()

        self.apply_thres()
        self.draw()
        self.repaint()

    def set_polarization(self,pol):
        self.pol = pol
        self.draw()
        self.repaint()

    def get_amp_threshold(self,amp_thres):
        self.amp_thres[self.baseline] = amp_thres
        self.apply_thres()
        self.draw()
        self.repaint()
        

    def get_rms_threshold(self,rms_thres):
        self.rms_thres[self.baseline] = rms_thres
        self.apply_thres()
        self.draw()
        self.repaint()
        
        
    def get_dropout_threshold(self,dropout_thres):
        self.dropout_thres[self.baseline] = dropout_thres
        self.flag_dropout()
        self.apply_thres()
        self.draw()
        self.repaint()


    def set_amp_threshold(self,amp_thres):
        self.amp_thres_entry.SetValue("%2.2f" % amp_thres)

    def set_rms_threshold(self,rms_thres):
        self.rms_thres_entry.SetValue("%2.2f" % rms_thres)

    def set_dropout_threshold(self,dropout_thres):
        self.dropout_thres_entry.SetValue("%2.2f" % dropout_thres)

    def apply_thres(self):
        bl = self.baseline
                
        uv.flg[bl][:,:] = 1
        uv.flg[bl][:,:] *= uv.dflg[bl]
        
        med_amp_rr = self.clean_amp[0]
        med_amp_ll = self.clean_amp[1]

        self.amp_thres_max[bl,0] = self.amp_thres[bl]*med_amp_rr
        self.amp_thres_max[bl,1] = self.amp_thres[bl]*med_amp_ll

        uv.flg[bl][:,:] *= np.where(uv.amp[bl][:,:,0]/med_amp_rr>self.amp_thres[bl],0,1)
        uv.flg[bl][:,:] *= np.where(uv.amp[bl][:,:,1]/med_amp_ll>self.amp_thres[bl],0,1)

        med_rms_rr = self.clean_rms[0]
        med_rms_ll = self.clean_rms[1]

        print 'rms=',self.rms_thres[bl]
        uv.flg[bl][:,:] *= np.where(uv.err[bl][:,:,0]/med_rms_rr>self.rms_thres[bl],0,1)
        uv.flg[bl][:,:] *= np.where(uv.err[bl][:,:,1]/med_rms_ll>self.rms_thres[bl],0,1)

    def apply_IF_thres(self,IF_thres):
        
        bl = self.baseline
        pol = self.pol

        # Clear old flags and apply dropout flag
        uv.flg[bl][:,:] = 1
        uv.flg[bl][:,:] *= uv.dflg[bl]
        
        IF = []
        for i in np.arange(8):
            a = IF_thres[i,pol,bl,:]
            IF.append(np.polyval(a,np.arange(451)))

        IF = np.concatenate(IF)
        IF = np.where(IF>self.amp_thres_max[bl,pol],self.amp_thres_max[bl,pol],IF)

        uv.flg[bl][:,:] *= np.where(uv.amp[bl][:,:,pol]-IF>0,0,1)

        self.draw()
        self.repaint()


    def draw(self):
        
        global bp_window
        
        bl = self.baseline
        pol = self.pol
        print bl

        print uv.amp[bl].shape,uv.flg[bl].shape
        if not hasattr(self,'subplot'):

            self.subplot = self.figure.add_subplot(111)
            print uv.amp[bl][:,:,pol].shape,uv.flg[bl].shape
            self.imshow = self.subplot.imshow(uv.amp[bl][:,:,pol]*uv.flg[bl],aspect='auto')
        else:
            self.imshow.set_data(uv.amp[bl][:,:,pol]*uv.flg[bl])

        vmin = np.min(uv.amp[bl][:,:,pol]*uv.flg[bl])
        vmax = np.max(uv.amp[bl][:,:,pol]*uv.flg[bl])
        self.imshow.set_clim(vmin=vmin,vmax=vmax)

        bp_window.draw()


    def repaint(self):

        global bp_window

        print self.baseline
        self.canvas.draw()

        bp_window.repaint()


class BPass_Window(wx.Window):
    
    def __init__(self, *args, **kwargs):
        wx.Window.__init__(self, *args, **kwargs)

        self.parent = self.GetParent()
        self.baseline = 1
        self.pol = 0

        nbas = len(uv.amp)        
        self.IF_thres = np.zeros((8,2,nbas,2),'f')

        self.figure = Figure(figsize=(4,3),dpi=100)
        self.canvas = FigureCanvasWxAgg(self,-1,self.figure)

        main_box = wx.BoxSizer(wx.HORIZONTAL)
        main_box.Add(self.canvas, flag=wx.EXPAND, proportion=1)
        self.SetSizer(main_box)

        self.draw()

    def get_IF_thres(self,bl,pol):
        amp = np.transpose(uv.amp[bl][:,:,pol])
        msk = np.transpose(uv.flg[bl])
        amp = ma.array(amp,mask=(1-msk))

        medfilt = ma.median(amp,axis=1)

        w = ma.std(amp,axis=1)
        w = np.where(w==0,1e4,w)
        w = 1/(w+1e-2)

        bp = []
        IF = []

        for i in np.arange(8):
            xf = np.where(medfilt[i*451:i*451+451]!=0)[0]            
            nxf = len(xf)

            if nxf>20:
                spl = UnivariateSpline(xf,medfilt[xf+i*451],w=w[xf+i*451],s=1)
                bp.append(spl(np.arange(451)))
                a = np.polyfit(xf,medfilt[xf+i*451],1,w=w[xf+i*451])

                if i==0:
                    a[1] = a[1]*1.5 + 2.5e-4 
                else:
                    a[1] = a[1]*1.5 + 2.5e-4 

                self.IF_thres[i,pol,bl,:] = a
 
                IF.append(np.polyval(a,np.arange(451)))
               
            elif nxf==0:
                bp.append(np.zeros(451))
                self.IF_thres[i,:,bl] = 0.0
                IF.append(np.zeros(451))
                
            else:
                spl = np.median(medfilt[xf+i*451])
                bp.append(np.ones(451)*spl)
                self.IF_thres[i,1,bl] = ma.median(medfilt[xf+i*451])*3.0
                self.IF_thres[i,0,bl] = 0
                IF.append(np.ones(451)*self.IF_thres[i,1,bl])
                 
        bp = np.concatenate(bp)
        IF = np.concatenate(IF)

        return bp, IF, amp, medfilt 

    def draw(self):
        bl = self.baseline
        pol = self.pol

        print "BP ",bl

        bp = [None,None]
        IF = [None,None]
        amp = [None,None]
        medfilt = [None,None]
        
        bp[0],IF[0],amp[0],medfilt[0] = self.get_IF_thres(bl,0)
        bp[1],IF[1],amp[1],medfilt[1] = self.get_IF_thres(bl,1)

        msk = np.transpose(uv.flg[bl])


        if not hasattr(self,'subplot'):
            self.subplot = self.figure.add_subplot(111)
            self.imshow = self.subplot.plot(amp[pol]*msk,',b')
            self.imshow = self.subplot.plot(medfilt[pol],'-g')
            self.imshow = self.subplot.plot(bp[pol],'-r')
            self.imshow = self.subplot.plot(IF[pol],'-k')
#            self.imshow = self.subplot.plot(IF_max,'--r')
#            self.imshow = self.subplot.plot(bp*.75,'--r')
        else:
            self.subplot.clear()
            self.imshow = self.subplot.plot(amp[pol]*msk,',b')
            self.imshow = self.subplot.plot(medfilt[pol],'-g')
            self.imshow = self.subplot.plot(bp[pol],'-r')
            self.imshow = self.subplot.plot(IF[pol],'-k')
#            self.imshow = self.subplot.plot(bp*1.25,'--r')
#            self.imshow = self.subplot.plot(bp*.75,'--r')

    def set_baseline(self,bl):
        self.baseline = bl
        self.draw()
        self.repaint()


    def repaint(self):
        self.canvas.draw()



class App(wx.App):
    def OnInit(self):

        self.frame1 = RFI_Frame(parent=None,title='RFI removal tool',size=(1400,850))
        self.frame1.Show()
        return True

if __name__=='__main__':

    fits_file = sys.argv[1]

    uv = RFI.read_fits(fits_file,progress=1)
    bl = 0

    app = App()
    app.MainLoop()

