import numpy as np
import RFI_fits as RFI
import smooth as sm

import pylab as plt

import cPickle as pic
import sys

global uv,baseline

def get_clean_level(bl,thres=90.0):
    cb = [(1000,1300),(1400,1650),(2200,2500),(3000,3150),(3250,3400)] 
        
    cb_index = []
    for i in np.arange(len(cb)):
        cb_index.append(np.arange(cb[i][0],cb[i][1]))

    cb_index = np.concatenate(cb_index)
    amp0 = np.median(uv.amp[bl][:,cb_index,0],axis=1)
    amp1 = np.median(uv.amp[bl][:,cb_index,1],axis=1)

    thres0 = np.percentile(amp0,thres)
    thres1 = np.percentile(amp1,thres)

    return amp0, amp1, thres0, thres1


def flag_dropout(bl,thres=90.0):

# Get threshold levels

    amp0, amp1, thres0,thres1 = get_clean_level(bl)

# Clean bits of spectrum

    uv.dflg[bl] = np.where((amp0[:,np.newaxis]<thres0/3),0,1) 
    uv.dflg[bl][:,:] *= np.where((amp1[:,np.newaxis]<thres1/3),0,1) 


def smooth(e,sig,ndx=25,apply_flags=False): 
    x = np.arange(-ndx,ndx+1)
    y = x[:,np.newaxis]
    r2 = x**2 + y**2
    g = np.exp(-r2/2/sig**2)
    
    w   = 1./e**2 * uv.dflg[baseline] 

    if apply_flags:
        w *= uv.flg[baseline]

    smo = sm.weighted(e,w,g)
        
    return smo

def apply_thres(amp_smooth_rr, rms_smooth_rr, amp_smooth_ll, rms_smooth_ll, amp_thres, rms_thres):
    bl = baseline
                
    uv.flg[bl][:,:] = 1
    uv.flg[bl][:,:] *= uv.dflg[bl]
        
    uv.flg[bl][:,:] *= np.where(uv.err[bl][:,:,0]/rms_smooth_rr>rms_thres[bl],
0,1)
    uv.flg[bl][:,:] *= np.where(uv.err[bl][:,:,1]/rms_smooth_ll>rms_thres[bl],
0,1)
    med_amp_rr = np.median(uv.amp[bl][:,:,0]*uv.flg[bl])
    med_amp_ll = np.median(uv.amp[bl][:,:,1]*uv.flg[bl])
    uv.flg[bl][:,:] *= np.where(uv.amp[bl][:,:,0]/med_amp_rr>amp_thres[bl],0,1)
    uv.flg[bl][:,:] *= np.where(uv.amp[bl][:,:,1]/med_amp_ll>amp_thres[bl],0,1)


##############################################################################
#
#  Main routine
#

# Get input filenames

in_path = sys.argv[1]
thres_file = sys.argv[2]


# Generate directory paths from input filename

data_dir = '/'.join(in_path.split('/')[:-1])
data_file = in_path.split('/')[-1]

flag_dir = data_dir+'/flags/'
html_dir = data_dir+'/html/'
image_dir = html_dir+'images/'


# Read in threshold setting pic file

pf = open(thres_file,'rb')

smo_val = pic.load(pf)
rms_thres = pic.load(pf)
amp_thres = pic.load(pf)
pf.close()


# Read in uvdata file to uv object

uv = RFI.read_fits(in_path,progress=0)

sig = 10.0

nbas = len(uv.amp)

levs0 = []
levs1 = []

for bl in np.arange(nbas):
    baseline = bl
    a1,a2 = uv.base_name[uv.bl[bl]]

    flag_dropout(bl)

    rms_smooth_rr = smooth(uv.err[bl][:,:,0],sig)
    amp_smooth_rr = smooth(uv.amp[bl][:,:,0],sig)
    rms_smooth_ll = smooth(uv.err[bl][:,:,1],sig)
    amp_smooth_ll = smooth(uv.amp[bl][:,:,1],sig)

    apply_thres(amp_smooth_rr, rms_smooth_rr, amp_smooth_ll, rms_smooth_ll, amp_thres, rms_thres)

    rms_smooth_rr = smooth(uv.err[bl][:,:,0],sig, apply_flags=True)
    amp_smooth_rr = smooth(uv.amp[bl][:,:,0],sig, apply_flags=True)
    rms_smooth_ll = smooth(uv.err[bl][:,:,1],sig, apply_flags=True)
    amp_smooth_ll = smooth(uv.amp[bl][:,:,1],sig, apply_flags=True)

    apply_thres(amp_smooth_rr, rms_smooth_rr, amp_smooth_ll, rms_smooth_ll, amp_thres, rms_thres)
    
    # Collect clean levels to allow final clip in AIPS 

    amp0, amp1, l0, l1 = get_clean_level(bl)
    levs0.append(l0)
    levs1.append(l1)


    # Make plot of cleaned RFI

    plot_file = image_dir+data_file[:-5]+('_%s_%s.png' % (a1,a2))

    plt.figure(1,figsize=(20,10))
    plt.clf()
    plt.imshow(uv.amp[bl][:,:,0]*uv.flg[bl],aspect='auto')
    plt.savefig(plot_file)


# Write out flag table

uv.write_flag_table(flg_dir=flag_dir)


# Write out pic file of levels

lev_file = flag_dir+'Flags_%s.pic' % uv.source

pf = open(lev_file, 'wb')
pic.dump(levs0,pf)
pic.dump(levs1,pf)
pf.close()
