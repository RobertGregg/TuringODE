using DifferentialEquations, ModelingToolkit, DiffEqParamEstim
using Turing, Distributions, StatsPlots
using CSV, DataFrames

include("mcmcODE.jl")

#=
1. Set up the ODE
2. Import and process the data
3. Transform the ODE output to match the data
4. Generate a loss function
5. Run the MCMC algorithm + Postprocessing
=#


###############################################################
                    # 1. Set up the ODE
###############################################################

#Constants in the model
cellVol = 3e-12 #Cell Volume (liters)
Na = 6.02e23 #Avagadro's number

#Defining non zero species in the model
m2c(molecule) = @. 1e9*molecule/(cellVol*Na) #Converts molecules to nM

#Initial Conditions
intConditions = (
cGAS = m2c(1e3),
DNA = m2c(1e3),
Sting = m2c(1e3),
cCAMP = 0.0,
IRF3 = m2c(1e4),
IFNbm = 0.0,
IFNb = 0.0,
STAT = 0.0,
SOCSm = 0.0,
IRF7m = 0.0,
TREX1m = 0.0,
IRF7 = 0.0,
TREX1 = 0.0)

#Algebraic Equations (mass balances)
cGAStot, Stingtot, IRF3tot = (
intConditions[:cGAS],
intConditions[:Sting],
intConditions[:IRF3])

## Define the ODE model
@parameters t k1f k1r k3f k3r k4f kcat5 Km5 k5r kcat6 Km6 kcat7 Km7 kcat8 Km8 k8f k9f k10f1 k10f2 k11f k12f k13f k6f kcat2 Km2 τ4 τ6 τ7 τ8 τ9 τ10 τ11 τ12 τ13
@variables cGAS(t) DNA(t) Sting(t) cGAMP(t) IRF3(t) IFNβm(t) IFNβ(t) STAT(t) SOCSm(t) IRF7m(t) TREX1m(t) IRF7(t) TREX1(t)
@derivatives D'~t

#Write the equations for the states
eqs= [
  D(cGAS) ~ -k1f*cGAS*DNA + k1r*(cGAStot - cGAS)
  D(DNA) ~ -k1f*cGAS*DNA + k1r*(cGAStot - cGAS) - kcat2*TREX1*DNA / (Km2 + DNA)
  D(Sting) ~ -k3f*cGAMP*Sting + k3r*(Stingtot - Sting)
  D(cGAMP) ~ k4f*(cGAStot - cGAS) - k3f*cGAMP*Sting + k3f*(Stingtot - Sting) - τ4*cGAMP
  D(IRF3) ~ -kcat5*IRF3*(Stingtot - Sting) / (Km5 +IRF3) + k5r*(IRF3tot - IRF3)
  D(IFNβm) ~ kcat6*(IRF3tot - IRF3) / (Km6 + (IRF3tot - IRF3)) + k6f*IRF7 - τ6*IFNβm
  D(IFNβ) ~ kcat7*IFNβm / (Km7 + IFNβm) - τ7*IFNβ
  D(STAT) ~ kcat8*IFNβ / (Km8 + IFNβ) * 1.0/(1.0+k8f*SOCSm) - τ8*STAT
  D(SOCSm) ~ k9f*STAT - τ9*SOCSm
  D(IRF7m) ~ k10f1*STAT + k10f2*IRF7 - τ10*IRF7m
  D(TREX1m) ~ k11f*STAT - τ11*TREX1m
  D(IRF7) ~ k12f*IRF7m - τ12*IRF7
  D(TREX1) ~ k13f*TREX1m - τ13*TREX1]

#Generate a function that evaluates the ODE
de = ODESystem(eqs)
f = ODEFunction(de,[cGAS, DNA, Sting, cGAMP, IRF3, IFNβm, IFNβ, STAT, SOCSm, IRF7m, TREX1m, IRF7, TREX1],[ k1f, k1r, k3f, k3r, k4f, kcat5, Km5, k5r, kcat6, Km6, kcat7, Km7, kcat8, Km8, k8f, k9f, k10f1, k10f2, k11f, k12f, k13f, k6f, kcat2, Km2, τ4, τ6, τ7, τ8, τ9, τ10, τ11, τ12, τ13])

#Contruct the ODE Problem
#Time to run the simulation
tspan = (0.0, 48.0)
#Initial guesses for the parameters
pNames = (:k1f, :k1r, :k3f, :k3r, :k4f, :kcat5, :Km5, :k5r, :kcat6, :Km6, :kcat7, :Km7, :kcat8, :Km8, :k8f, :k9f, :k10f1, :k10f2, :k11f, :k12f, :k13f, :k6f, :kcat2, :Km2, :τ4, :τ6, :τ7, :τ8, :τ9, :τ10, :τ11, :τ12, :τ13)

p = [2.6899, 4.8505, 0.0356, 7.487, 517.4056, 22328.3852, 11226.3682,0.9341,
         206.9446, 10305.461, 47639702.95,3.8474, 13.006, 78.2048, 0.0209,
         0.0059, 0.001, 0.0112, 0.001, 99.9466, 15.1436,0.0276, 237539.3249,
         61688.259, 0.96, 1.347, 12242.8736,1.2399, 1.5101, 0.347, 0.165, 6.9295,
         0.0178]


#Get the values for the initial condtions
u0 = collect(intConditions)

#Contruct the problem
prob = ODEProblem(f,u0,tspan,p)

#Choose an ODE algorithm and solve for the solution
alg = Rodas5()
sol = solve(prob,alg)

###############################################################
                    # 2. Import the data
###############################################################

#Read CSV in
data = CSV.read("NormData.csv",missingstring = "-")

#Two seperate error functions are used, one comparing normalized simulations and data, and the other comparing desired peaks (max/min)

#Define desired peaks
desiredScale = m2c([0.8*1e3,eps(),0.8*1e3, 1e5, 3e3,100.0,1e5,1e4,
                    30.0,30.0,30.0,1e4,1e4])

#Transform dataframe into Array of Arrays
timePoints = [data[data.Experiment .==i,2] for i ∈ unique(data.Experiment)]
experiments = [data[data.Experiment .==i,3:end] for i ∈ unique(data.Experiment)]

#Convert into matrices and vectors
timePoints = convert.(Vector,timePoints)
experiments = convert.(Matrix,experiments)

#Scale data by the desired value
dataIndex = [6,7,10,9,11,5,3]

scaledExperiments = [ desiredScale[dataIndex]'.*exp for exp in experiments]

###############################################################
        # 3. Transform the ODE output to match the data
###############################################################

#Depending on the data you might want to convert your ODE solution to match the data (convert to log fold change, measuring different species in mass balance, etc)

#Example expIdx
#expIdx = @. !ismissing(experiments[1][1,:])

function Observables(sol,intConditions,dataIndex,t)
  obversedSpecies = sol(t)[dataIndex,:]

  if isa(t,Number)
    #Have data for activated protiens, ode returns inactive
    obversedSpecies[end] = intConditions[:Sting] - obversedSpecies[end]
    obversedSpecies[end-1] = intConditions[:IRF3] - obversedSpecies[end-1]
  else
    obversedSpecies[end,:] = intConditions[:Sting] .- obversedSpecies[end,:]
    obversedSpecies[end-1,:] = intConditions[:IRF3] .- obversedSpecies[end-1,:]
  end

  #What species are being measured?
  return obversedSpecies
end

function heuristic(sol,desiredScale)
  peaksModel = maximum(sol[:,:],dims=2)
  peaksModel[[1,3,5]] = minimum(sol[[1,3,5],:],dims=2)
  h = @. (1.0-peaksModel/desiredScale)^2
  h[2] = 0.0
  return sum(h)
end
#What parameters are we changing?
#Need to pass the parameter values and the indicies of the parameters you want to change
parVary = (:k1f, :k1r, :k3f, :k3r, :k4f, :kcat5, :Km5, :k5r, :kcat6, :Km6, :k11f, :k13f, :kcat2, :Km2, :τ7, :kcat7, :Km7, :kcat8, :Km8, :k8f, :k9f, :k10f1, :k10f2, :k12f, :k6f)

#Number of parameters in the model to vary
parNum = length(parVary)

parVaryIdx = findall(par -> par ∈ parVary, pNames)

parChange = NamedTuple{parVary}(parVaryIdx)

###############################################################
                    # 4. Run the MCMC algorithm
###############################################################

#Provide any prior knownledge for the parameters
priors = fill(Uniform(0,1),parNum)

#include("mcmcODE.jl")
result = mcmcODE(prob,alg,timePoints,experiments,priors,parVary,parVaryIdx;num_samples=1000)

###############################################################
                # 5. Postprocessing & Plotting
###############################################################

#Summary of the MCMC results
#describe(result)

#Save the parameter chains and information about the MCMC run
chainParameters = DataFrame(result,:parameters)
CSV.write("Parameters.csv",chainParameters)

chainInternals = DataFrame(result,:internals)
CSV.write("Internals.csv",chainInternals)


#Make a plot of the chains
chainsPlot = plot(result)
savefig(chainsPlot,"ChainsODE.pdf")


#Plot the data against the simulation

dataPlots = Vector(undef,length(dataIndex))
#loop through the data
for (i,protein) in enumerate(names(data)[3:end])
  dataPlots[i] = @df data plot(:Time,data[!,protein].*desiredScale[dataIndex[i]],
                 group=:Experiment,title=protein,markershape=:auto)
end

plot(dataPlots...,legend=false)


#Get the best parameter through the lowest log Posterior
bestPar = copy(p)
bestParIdx = argmax(chainInternals.lp)

#Why are the parameters out of order, why do you hate my Turing.jl?
for par in names(chainParameters)
  if par ∈ pNames
  bestPar[findfirst(x-> x==par,pNames)] = chainParameters[bestParIdx,par]
  end
end

newProb = remake(prob, p=bestPar)
newSol = solve(newProb,alg)

t = range(tspan[1],tspan[2],length=100)
obs = Observables(sol,intConditions,dataIndex,t)

plot!(t,obs',layout=length(dataIndex),legend=false,framestyle=:box)
