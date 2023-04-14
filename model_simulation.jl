# Updated for final use 210427, changed location of refractory
# period enforcement to just before stimulation

using DataStructures
using Distributions
using Statistics
using StatsBase
using LinearAlgebra
using JSON

function myStr(numb, d)
    """
    converts a number into a string that works in a file name.
    takes any type of number, numb, and rounds to d decimal places
    if d==0, then numb if rounded to an Int
    this number is cast into a string
    the decimal point is replaced by p
    and a negative sign is replaced by n
    returns a string
    """
    if d == 0
        numb = Int(round(numb))
    else
        numb = round(numb,digits=d)
    end
    s = string(numb)
    s = replace(s, Pair(".","p"))
    s = replace(s, Pair("-","n"))
    return s
end

function UpdateStimList(stimTs, stimNs, N)
    si = []
    stimT = 0
    if length(stimTs) > 0 # update stimT if possible
        # define stimulation time
        stimT = stimTs[1]

        # update stim neurons, si, if possible
        stimN = stimNs[1]
        if stimN > 0 # update si if nonzero
            si = Array{Int,1}(undef,stimN)
            StatsBase.self_avoid_sample!(1:N,si)
        end
        
        # update stim Ts and Ns lists
        stimTs = copy(stimTs[2:end])
        stimNs = copy(stimNs[2:end])
    end
    return stimT, si, stimTs, stimNs
end

function mysim(J, vext, g, N, T, 
        Iin, Din, 
        Wprepost, ks, Wextprepost, kexts, dt, dtsave, delay, tref, stimTs, stimNs, nustim, stimWin)
    
    # model constants
    Ne = Int(N*.8)
    tau = 20.0
    thresh = 20.0
    
    # time steps and delay/refractory queues
    ts = dt:dt:T
    lts = Int(round(T/dt))
    t_to_spike = zeros(N)
    spikeDelayQueue = Queue{Array{Int}}()
    for _ in 1:Int(round(delay/dt))
        enqueue!(spikeDelayQueue,copy([]))
    end
    println("T",T," dtsave",dtsave)
    
    # storage tools
#     vstor = zeros(N,Int(T/dtsave))
#    pvstor = zeros(3000,Int(T/dtsave))
    fstor = zeros(Int(T/dtsave)+1)
#    fstor1000 = zeros(1000, Int(T/dtsave)+1)
#     vstor = zeros(N,Int(T/dtsave)+1-4900)
    alltimes = []
    allspikes = []
    println(length(fstor))
    curti = 1
    curtint = 1
    
    #voltage initialization
    v = (10.0 * randn(N)) .+ 0.0
    
    #external input
    dext = Poisson(Ne*vext*dt)
    
    # stimulus
    StimNow = false
    nustimd = Poisson(dt*nustim)
    stimT, si, stimTs, stimNs = UpdateStimList(stimTs, stimNs, N)
    
    for i in 1:lts
        t = i*dt
        # update dynamics of v
        for n in 1:N
            if !(t_to_spike[n] > 0.0)
                dvdt = -copy(v[n])/tau
                #dvdt += (Iin/tau) + (sqrt(Din/tau)/sqrt(dt))*randn()
                v[n] += dvdt * dt
            end
        end
        
        # apply recurrent spikes
        spikearrive = dequeue!(spikeDelayQueue)
        for n in spikearrive
            if n > Ne
                v[Wprepost[n,1:ks[n]]] .+= -g*J
            else
                v[Wprepost[n,1:ks[n]]] .+= J
            end
        end
        
        # apply external spikes, poisson
        extspikearrive = sample(1:Ne, rand(dext), replace=true, ordered=true)
        for n in extspikearrive
            v[Wextprepost[n,1:kexts[n]]] .+= J
        end
        
        # enforce refractory period before spiking is assessed and before stimulation, after input spikes
        for n in 1:N
            if t_to_spike[n] > 0.0
                v[n] = 0.0
                t_to_spike[n] -= dt
            else
                t_to_spike[n] = -1.0
            end
        end
        
        # apply stimulus spikes to all, including refractory neurons
        if t >= stimT && t < stimT+stimWin
            StimNow = true
            if length(si) > 0
                for i in 1:length(si)
                    if rand(nustimd,1)[1] == 1
                        v[si[i]] = thresh + 1
                    end
                end
            end
        else
            if StimNow
                stimT, si, stimTs, stimNs = UpdateStimList(stimTs, stimNs, N)
            end
            StimNow = false
        end
        
        #v = copy(v)
            
        # collect threshold spikes and reset
        spikedepart = []
        for n in 1:N
            if v[n] >= thresh
                v[n] = 0.0
                t_to_spike[n] = tref
                push!(spikedepart, copy(n))
            end
        end
        enqueue!(spikeDelayQueue, copy(spikedepart))
        
        # save
        
        append!(allspikes, copy(spikedepart))
        append!(alltimes, t*ones(length(spikedepart))) 
        if t > (curti+1)*dtsave
            curti += 1
           #pvstor[:,curti] = fit(Histogram, v, (LinRange(-10,20,3001))).weights
        end
        if t > (curtint+1)
            curtint += 1
        end
        fstor[curti] += length(spikedepart)
#        fstor1000[filter(x -> x <= 1000, spikedepart), curti] .+= 1
#         if curti>4900
#             vstor[:,curti-4900] = v
#         end
    end
    return fstor, allspikes, alltimes #pvstor , fstor1000#, vstor
end

function myBrunelA(g,v,J,Ce,N,T, delay, tref, dt, dtsave, date, stimTs, stimNs, nustim, stimWin)

    postInfo = "_delay"*myStr(delay,3)*"_tref"*myStr(tref,3)

    theta, tau = 20.0, 20.0

    Ne = N*.8
    eps = Ce/Ne
    Ne = Int(Ne)
    Ce = Int(Ce)
    C = Int(eps*N)
    Ci = C - Ce
    
    #d = Binomial(N, eps)
    #ks = rand(d, N)
    #Wprepost = fill(0,N,maximum(ks))
    #for n in 1:N
    #    postInds = sample(1:N,ks[n], replace=false, ordered=true)
    #    Wprepost[n,1:ks[n]] = copy(postInds)
    #end

    ks = zeros(Int,N)
    Wprepost = fill(0,N,Int(2*(Ce+Ci)))
    for posti in 1:N
        preIndsE = sample(1:Ne,Int(Ce), replace=false, ordered=true)
        for piei in 1:Int(Ce)
            pie = preIndsE[piei]
            ks[pie] = ks[pie]+1
            Wprepost[pie,ks[pie]] = posti
        end
        preIndsI = sample(Ne+1:N,Int(Ci), replace=false, ordered=true)
        for piii in 1:Int(Ci)
            pii = preIndsI[piii]
            ks[pii] = ks[pii]+1
            Wprepost[pii,ks[pii]] = posti
        end
    end

    #dext = Binomial(N, eps)
    #kexts = rand(dext, Ne)
    #Wextprepost = fill(0,Ne,maximum(kexts))
    #for n in 1:Ne
    #    postInds = sample(1:N,kexts[n], replace=false, ordered=true)
    #    Wextprepost[n,1:kexts[n]] = copy(postInds)
    #end

    kexts = zeros(Int,N)
    Wextprepost = fill(0,Ne,Int(2*(Ce+Ci)))
    for posti in 1:N
        preIndsE = sample(1:Ne,Int(Ce), replace=false, ordered=true)
        for piei in 1:Int(Ce)
            pie = preIndsE[piei]
            # adding another post to the preind list
            kexts[pie] = kexts[pie]+1
            Wextprepost[pie,kexts[pie]] = posti
        end
        #preIndsI = sample(Ne+1:N,Int(Ci), replace=false, ordered=true)
        #for piii in 1:Int(Ci)
        #    pii = preIndsI[piii]
        #    # adding another post to the preind list
        #    kexts[pii] = kexts[pii]+1
        #    Wextprepost[pii,kexts[pii]] = posti
        #end
    end

    vthr = theta/(J*Ce*tau)
    vext = v*vthr

    Iin = vext*J*Ce*tau
    Din = (J^2)*Ce*vext*tau
    println("Iin and Din ", Iin, " , ",Din)

    @time begin
    fs, allspikes, alltimes = mysim(J, vext, g, N, T, 
        Iin, Din, 
        Wprepost, ks, Wextprepost, kexts, 
        dt, dtsave, 
        delay, tref, stimTs, stimNs, nustim, stimWin)
    end

    open(date*"_N"*myStr(N,0)*"_v"*myStr(v,6)*"_g"*myStr(g,6)*"_dts"*myStr(dtsave,6)*"_dt"*myStr(dt,6)*postInfo*".json","w") do f
        JSON.print(f, fs)
    end
    open(date*"_StimTimes"*"_N"*myStr(N,0)*"_v"*myStr(v,6)*"_g"*myStr(g,6)*"_dts"*myStr(dtsave,6)*"_dt"*myStr(dt,6)*postInfo*".json","w") do f
        JSON.print(f, stimTs)
    end
    #open(date*"_PVs"*"_N"*myStr(N,0)*"_v"*myStr(v,6)*"_g"*myStr(g,6)*"_dts"*myStr(dtsave,6)*"_dt"*myStr(dt,6)*postInfo*".json","w") do f
    #    JSON.print(f, pvstor)
    #end
#     open(date*"_fstor1000_N"*myStr(N,0)*"_v"*myStr(v,4)*"_g"*myStr(g,4)*"_dts"*myStr(dtsave,6)*"_dt"*myStr(dt,6)*postInfo*".json","w") do f
#         JSON.print(f, fs1000)
#     end
     open(date*"_spiketimes_N"*myStr(N,0)*"_v"*myStr(v,6)*"_g"*myStr(g,6)*"_dts"*myStr(dtsave,6)*"_dt"*myStr(dt,6)*postInfo*".json","w") do f
         JSON.print(f, alltimes)
     end
     open(date*"_spikes_"*"_N"*myStr(N,0)*"_v"*myStr(v,6)*"_g"*myStr(g,6)*"_dts"*myStr(dtsave,6)*"_dt"*myStr(dt,6)*postInfo*".json","w") do f
         JSON.print(f, allspikes)
     end
	
    scratch_folder = "../../../../../scratch/users/bensonb/"
    open(scratch_folder*date*"_N"*myStr(N,0)*"_v"*myStr(v,6)*"_g"*myStr(g,6)*"_dts"*myStr(dtsave,6)*"_dt"*myStr(dt,6)*postInfo*".json","w") do f
        JSON.print(f, fs)
    end
    open(scratch_folder*date*"_StimTimes"*"_N"*myStr(N,0)*"_v"*myStr(v,6)*"_g"*myStr(g,6)*"_dts"*myStr(dtsave,6)*"_dt"*myStr(dt,6)*postInfo*".json","w") do f
        JSON.print(f, stimTs)
    end
    #open(scratch_folder*date*"_PVs"*"_N"*myStr(N,0)*"_v"*myStr(v,6)*"_g"*myStr(g,6)*"_dts"*myStr(dtsave,6)*"_dt"*myStr(dt,6)*postInfo*".json","w") do f
    #    JSON.print(f, pvstor)
    #end
#     open(scratch_folder*date*"_fstor1000_N"*myStr(N,0)*"_v"*myStr(v,4)*"_g"*myStr(g,4)*"_dts"*myStr(dtsave,6)*"_dt"*myStr(dt,6)*postInfo*".json","w") do f
#         JSON.print(f, fs1000)
#     end
     open(scratch_folder*date*"_spiketimes_N"*myStr(N,0)*"_v"*myStr(v,6)*"_g"*myStr(g,6)*"_dts"*myStr(dtsave,6)*"_dt"*myStr(dt,6)*postInfo*".json","w") do f
         JSON.print(f, alltimes)
     end
     open(scratch_folder*date*"_spikes_"*"_N"*myStr(N,0)*"_v"*myStr(v,6)*"_g"*myStr(g,6)*"_dts"*myStr(dtsave,6)*"_dt"*myStr(dt,6)*postInfo*".json","w") do f
         JSON.print(f, allspikes)
     end
end
