#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# 210120 Creating
# 210427 Simplifying for final use
# 210901 Synchronizing with theory and simplify for final use

@author: bensonb
"""
import numpy as np
import scipy.stats
import pickle
import os

def f0_default(xs):
    ''' xs is a linearly spaced 1d numpy array of support for the initialized output distribution f0'''
    dx = xs[1]-xs[0]
    fs = scipy.stats.norm.pdf(xs,loc=15,scale=5)-scipy.stats.norm.pdf(xs,loc=25,scale=5)
    resetInd = int(np.interp(0,xs,np.arange(len(xs))))
    fs[:resetInd] = 0
    return fs*.999/(np.sum(fs)*dx)

def fvstim_default(t):
    return 0.0


def UpdateNuOld(nu_t, nu_old):
    nu_t_minus_D = nu_old[0]
    nu_old[:-1] = nu_old[1:]
    nu_old[-1] = nu_t
    return nu_t_minus_D, nu_old

def UpdateState(nu_t_minus_D_excitatory, nu_t_minus_D_inhibitory, vstim, state, params):
    
    dt, vext, J, g, Ce, Ci, tau, vs, dv, Vreset, resetInd, dtsave, dtsavepv, vskip = params
    ps, Bs = state


    ############## Time Shift dynamics ################################

    Bs_shift = np.empty(len(Bs))
    Bs_shift[:-1] = Bs[1:]
    Bs_shift[-1] = 0
    Bs_shift[0] += np.copy(Bs[0])
    dBs_shift = Bs_shift - Bs


    ############## L-leaky dynamics ################################

    edt = np.exp(dt/tau)
    csum = np.cumsum(ps)
    csum_support = vs + (dv/2.0)
    l_csum = np.interp((vs - (dv/2.0)) * edt, csum_support, csum)
    u_csum = np.interp((vs + (dv/2.0)) * edt, csum_support, csum)
    ps_leak = u_csum - l_csum
    dp_leak = ps_leak - ps

    ############## input firing rate, jump process ##############

    pb = np.interp(vs - J,vs,ps)
    pa = np.interp(vs + g*J,vs,ps,right=0) # added right=0 on 210512
    ps_jump = np.copy(ps)
    ps_jump += (dt*nu_t_minus_D_excitatory*(Ce*(pb-ps))) + (dt*nu_t_minus_D_inhibitory*(Ci*(pa-ps))) # Recurrent input spikes
    ps_jump += dt*vext*Ce*(pb-ps) # External input spikes

    psthreshperffire = np.sum(Ce*ps - Ce*pb)
    ps_jump[-1] = ps_jump[-1] + dt*nu_t_minus_D_excitatory*psthreshperffire
    ps_jump[-1] = ps_jump[-1] + dt*vext*psthreshperffire

    dp_jump = ps_jump - ps

    ############## stimulation ##############

    ps_stim = (1 - (vstim*dt)) * ps
    Bs_stim = (1 - (vstim*dt)) * Bs

    ps_stim[-1] = ps_stim[-1] + (vstim*dt/dv)

    dp_stim = ps_stim - ps
    dBs_stim = Bs_stim - Bs


    #################### Update P(V,t), B(U,t) #########################

    ps = np.copy(ps + dp_leak + dp_jump + dp_stim)
    Bs = np.copy(Bs + dBs_shift + dBs_stim)

    ############## boundary conditions and update firing rate ##############

    dp_from_B = Bs[0]*(dt/dv)
    dB_to_p = -Bs[0]

    dB_from_p = ps[-1]*(dv/dt)
    dp_to_B = -ps[-1]

    ps[-1] += np.copy(dp_to_B)
    ps[resetInd] += np.copy(dp_from_B)
    Bs[-1] += np.copy(dB_from_p)
    Bs[0] += np.copy(dB_to_p)

    nu_t = np.copy(dB_from_p)

    return nu_t, (ps, Bs)

def CheckAndSummarize(summary,t,nu_t,state,params):
    dt, vext, J, g, Ce, Ci, tau, vs, dv, Vreset, resetInd, dtsave, dtsavepv, vskip = params
    savepv, norms, mus, firingRates, firingRatesNCounts, Negs, lastt, lastft = summary
    ps, Bs = state
    ShutDown = False
    
    currentt = int(t/dtsavepv)
    if currentt > lastt:
        lastt = currentt
        savepv[currentt,:] = ps[::vskip]
        print(currentt)

    currentft = int(t/dtsave)
    if currentft > lastft:
        lastft = currentft
        norms[currentft] = (np.sum(ps)*dv) + (np.sum(Bs)*dt)
        mus[currentft] = (np.sum(ps*vs)*dv) + (np.sum(Bs*Vreset)*dt)
    firingRates[currentft] += nu_t
    firingRatesNCounts[currentft] += 1

    # diagnostics and system shut down
    negJI = np.nonzero(ps<0)[0]
    if len(negJI):
        Negs += 1

    if np.sum(ps[0:int(resetInd/2)])*dv > 1e-2:
        print('monitor base shut off')
        ShutDown = True
    if Negs >= 100:
        print('monitor negatives shut off')
        ShutDown = True
    if nu_t > 10.0:
        print('monitor firing rate shut off')
        ShutDown = True
    return (savepv, norms, mus, firingRates, firingRatesNCounts, Negs, lastt, lastft), ShutDown

def groupOutputs(filename, state, summary, ShutDown, params, 
                 func_f0, 
                 func_fvstim):
    ps, Bs = state
    savepv, norms, mus, firingRates, firingRatesNCounts, Negs, lastt, lastft = summary
    dt, vext, J, g, Ce, Ci, tau, vs, dv, Vreset, resetInd, dtsave, dtsavepv, vskip = params
    
    ts = (np.arange(lastt+1) - .5) * dtsavepv
    valls = np.copy(vs)
    pvallsfinal = ps
    vs = vs[::vskip]
    dv = vs[1]-vs[0]
    vs = np.append(vs,vs[-1]+dv) - (dv/2)

    tsfire = np.arange(lastft) * dtsave

    groupPofV = (ts,vs,savepv[:len(ts)-1,:])#used to be (ts,vs,savepv[:lastt+1,:])
    firingRates = firingRates/firingRatesNCounts
    groupfRate = (tsfire, firingRates[:len(tsfire)], mus[:len(tsfire)])#used to be (tsfire, firingRates[:lastft], mus[:lastft])
    diagnostics = (Negs,
                   ShutDown,
                   norms[:len(tsfire)])
    pfinal = (valls, pvallsfinal)
    
    # save as pickle
    dictionary = {'dynamicFokkerPlank(ts,vs,pvs)': groupPofV,
                 'summary(ts,firingRates,mus)': groupfRate,
                 'diagnostics(Negs,ShutDown,norms)': diagnostics,
                 'finalFokkerPlank_highdef(vs,ps)': pfinal,
                 'all_f0(vs,ps)': (valls, func_f0(valls)),
                 'all_fvstim(ts,fs)': (tsfire, np.array([func_fvstim(t) for t in tsfire]))}
    pickle.dump(dictionary,open('.'.join((filename,'p')),'wb'))
    
    return groupPofV, groupfRate, diagnostics, pfinal

def model_details(J,g,vextOvthr,Ce,T,
                             dtsave,
                             dt, f0, 
                             fvstim, fvstim_inhibitory, 
                             J_I, g_I, vextOvthr_I,
                             initfactor):
    function2string = lambda f: 'None' if f is None else 'Custom'
    out = '_'.join(('J'+str(J),
                  str(J_I),
                  'g'+str(g),
                  str(g_I),
                  'v'+str(vextOvthr),
                  str(vextOvthr_I),
                  'Ce'+str(Ce),
                  'T'+str(T),
                  str(dtsave),
                  str(dt),
                  'f0'+function2string(f0),
                  'fvstim'+function2string(fvstim),
                  function2string(fvstim_inhibitory),
                  'init'+str(initfactor)))
    
    return out.replace('.','p')

def DFP(J,g,vextOvthr,Ce,T=1000,
                 verbose=True,dtsave=.1,
                 dt=0.01, f0 = None, 
                 fvstim = None, fvstim_inhibitory = None, 
                 J_I = None, g_I = None, vextOvthr_I = None,
                 initfactor=1.0,
                 SAVE_PATH=''):
    filename = model_details(J,g,vextOvthr,Ce,T,
                             dtsave,
                             dt, f0, 
                             fvstim, fvstim_inhibitory, 
                             J_I, g_I, vextOvthr_I,
                             initfactor)
    print(filename)
    if f0 is None:
        f0 = f0_default
    if fvstim is None:
        fvstim = fvstim_default
    # Time related constants
    t = 0.0
    lastt, lastft = -1, -1
    currentt, currentft = 0, 0
    dtsavepv = 1.0

    # Model related parameters
    # only tested with these all as None
    if J_I is None:
        J_I = J
    if g_I is None:
        g_I = g
    if vextOvthr_I is None:
        vextOvthr_I = vextOvthr
        
    fvstim_excitatory = fvstim
    if fvstim_inhibitory is None:
        fvstim_inhibitory = lambda x:0.0
        
    tau, theta = 20., 20.
    tref, delay = 2.0, 1.5
    gamma = .25
    Ci = gamma*Ce
    vthr = theta/(J*Ce*tau)
    Vreset = 0

    # fokker-planck initialization
    vs = np.linspace(-20,20,40001) # one bin every .1, boundaries are between these v values
    resetInd = int(np.interp(Vreset,vs,np.arange(len(vs))))
    dv = vs[1]-vs[0]
    ps = f0(vs)

    # external population and stimulation
    vext = vextOvthr*vthr
    vext_I = vextOvthr_I*vthr


    # prepare to save
    vskip = 10
    savepv = np.zeros((int(T/dtsavepv),len(vs[::vskip])))
    mus = np.zeros(int(T/dtsave))
    firingRates = np.zeros(int(T/dtsave))
    firingRatesNCounts = np.zeros(int(T/dtsave))
    # diagnostics
    Negs, ShutBase = 0, False
    norms = np.zeros(int(T/dtsave))

    # refractory queue
    Bdt_norm = (1 - (np.sum(ps)*dv))
    #nu_old = Bdt_norm*np.ones(int(delay/dt))/int(tref/dt)

    Blen = int(tref/dt)
    Bs = (Bdt_norm/dt)*(np.ones(Blen)/Blen)
    nu_t = Bdt_norm/tref
    nu_old = initfactor*nu_t*np.ones(int(delay/dt))
    
    # useful grouping of parameters, state, and read-out
    nu_t_excitatory = np.copy(nu_t)
    params_excitatory = (dt, vext, J, g, Ce, Ci, tau, vs, dv, Vreset, resetInd, dtsave, dtsavepv, vskip)
    state_excitatory = (np.copy(ps), np.copy(Bs))
    nu_old_excitatory = np.copy(nu_old)
    summary_excitatory = (np.copy(savepv), np.copy(norms), np.copy(mus), 
                          np.copy(firingRates), np.copy(firingRatesNCounts), 
                          np.copy(Negs), np.copy(lastt), np.copy(lastft))
    
    
    nu_t_inhibitory = np.copy(nu_t)
    params_inhibitory = (dt, vext_I, J_I, g_I, Ce, Ci, tau, vs, dv, Vreset, resetInd, dtsave, dtsavepv, vskip)
    state_inhibitory = (np.copy(ps), np.copy(Bs))
    nu_old_inhibitory = np.copy(nu_old)
    summary_inhibitory = (np.copy(savepv), np.copy(norms), np.copy(mus), 
                          np.copy(firingRates), np.copy(firingRatesNCounts), 
                          np.copy(Negs), np.copy(lastt), np.copy(lastft))

    i = -1
    while t <= T-1:
        i += 1
        t += dt
        
        vstim_excitatory = fvstim_excitatory(t)
        vstim_inhibitory = fvstim_inhibitory(t)
        
        # keep track of firing rates: nu after the delay, D
        nu_t_minus_D_excitatory, nu_old_excitatory = UpdateNuOld(nu_t_excitatory, nu_old_excitatory)
        nu_t_minus_D_inhibitory, nu_old_inhibitory = UpdateNuOld(nu_t_inhibitory, nu_old_inhibitory)
        
        # update state of master equation
        nu_t_excitatory, state_excitatory = UpdateState(nu_t_minus_D_excitatory, nu_t_minus_D_inhibitory, vstim_excitatory, 
                                                        state_excitatory, params_excitatory) 
        
        nu_t_inhibitory, state_inhibitory = UpdateState(nu_t_minus_D_excitatory, nu_t_minus_D_inhibitory, vstim_inhibitory, 
                                                        state_inhibitory, params_inhibitory) 

        # saving and diagnostics
        summary_excitatory, ShutDown_excitatory = CheckAndSummarize(summary_excitatory, t, nu_t_excitatory, 
                                                                    state_excitatory, params_excitatory)
        summary_inhibitory, ShutDown_inhibitory = CheckAndSummarize(summary_inhibitory, t, nu_t_inhibitory, 
                                                                    state_inhibitory, params_inhibitory)
        if ShutDown_excitatory or ShutDown_inhibitory:
            break
############## post-processing: saving and plotting #############
# savepv, firingRates, firingRatesNCounts, mus, Negs, ShutBase, norms
    outs_excitatory = groupOutputs(os.path.join(SAVE_PATH,
                                                '_'.join(('excitatory',filename))),
                                   state_excitatory, summary_excitatory, ShutDown_excitatory, params_excitatory,
                                   f0, 
                                   fvstim_excitatory)
    outs_inhibitory = groupOutputs(os.path.join(SAVE_PATH,
                                                '_'.join(('inhibitory',filename))),
                                   state_inhibitory, summary_inhibitory, ShutDown_inhibitory, params_inhibitory,
                                   f0,
                                   fvstim_inhibitory)
    
    return outs_excitatory, outs_inhibitory
