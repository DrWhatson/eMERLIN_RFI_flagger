from AIPS import AIPS, AIPSDisk
from AIPSTask import AIPSTask, AIPSList
from AIPSData import AIPSUVData, AIPSImage, AIPSCat

import numpy as np
import cPickle as pic

userno = 1714
dsk = 2

AIPS.userno = userno

ant_no = {1:1, 2:2, 3:5, 4:6, 5:7, 6:8, 7:9}
nant = len(ant_no)

base_ant=[]
for i in np.arange(nant):
    for j in np.arange(i+1,nant):
        base_ant.append((ant_no[i+1],ant_no[j+1]))


#filelist = ['Flags_1020+6812.fits',
#            'Flags_1021+6815.fits',
#            'Flags_1034+6832.fits',
#            'Flags_0319+415.fits',
#            'Flags_1021+6803.fits',
#            'Flags_1022+6806.fits',
#            'Flags_1331+305.fits',
#            'Flags_1020+6806.fits',
#            'Flags_1021+6809.fits',
#            'Flags_1022+6812.fits',
#            'Flags_1407+284.fits']

filelist = ['Flags_1022+6749.fits']
#            'Flags_1331+305.fits',
#            'Flags_1407+284.fits']

flag_dir = '/scratch/nas_mberc2/SuperCLASS/Data/eMERLIN/2C_20150720/flags/'


fitld = AIPSTask('FITLD')
fitld.outdisk = dsk

tacop = AIPSTask('TACOP')

uvcop = AIPSTask('UVCOP')

clip = AIPSTask('CLIP')

#move = AIPSTask('MOVE')


for f in filelist:
    src = f.split('_')[-1].split('.')[0]
#    fitld.datain = flag_dir+f
#    outdata = AIPSUVData(src,'FG_TAB',dsk,1)
#    fitld.outdata = outdata
#    print "Reading %s to %s" % (f,src)
#    fitld.go()

#    tacop.inname = src
#    tacop.inclass = 'FG_TAB'
#    tacop.inseq = 1
#    tacop.indisk = 2
#    tacop.inext = 'FG'
#    tacop.outname = src
#    tacop.outclass = 'UVDATA'
#    tacop.outseq = 1
#    tacop.outdisk = 2
#    tacop.go()

#    uvcop.inname = src
#    uvcop.inclass = 'UVDATA'
#    uvcop.inseq = 1
#    uvcop.indisk = 2
#    uvcop.flagver = 2
#    uvcop.outname = src
#    uvcop.outclass = 'UVCOPY'
#    uvcop.outseq = 1
#    uvcop.outdisk = 2
#    uvcop.go()

    lev_file = flag_dir+f[:-4]+'pic'
    pf = open(lev_file,'rb')
    lev0 = pic.load(pf)
    lev1 = pic.load(pf)
    pf.close()

    nbas = 21

    clip.inname = src
    clip.inclass = 'UVCOPY'
    clip.inseq = 1
    clip.indisk = 2
    
    # Go through baselines and clip those visibilies that
    # still get through at > 4x the mean clean level 

    for ib in np.arange(1,nbas):
        ant1 = base_ant[ib][0]
        ant2 = base_ant[ib][1]

        clip.antennas[1] = ant1
        clip.antennas[2] = 0

        clip.baseline[1] = ant2
        clip.baseline[2] = 0

        clip.stokes = 'RR'
        clip.aparm[1] = float(lev0[ib]*4)
        clip.aparm[2] = float(lev0[ib]*4)
        clip.aparm[3] = 0.0
        clip.aparm[4] = 0.0
        clip.go()

        clip.stokes = 'LL'
        clip.aparm[1] = float(lev1[ib]*4)
        clip.aparm[2] = float(lev1[ib]*4)
        clip.aparm[3] = 0.0
        clip.aparm[4] = 0.0
        clip.go()

    
#    uvdata = AIPSUVData(src,'UVCOPY',dsk,1)
#    uvdata.rename(klass='UVDATA')

#    move.inname = src
#    move.inclass = 'UVCOPY'
#    move.inseq = 1
#    move.indisk = 2
#    move.outname = src
#    move.outclass = 'UVDATA'
#    move.outseq = 0
#    move.outdisk = 1
#    move.userid = 1710
#    move.go()
