#Importing Neccessary Packages
using DifferentialEquations,ModelingToolkit #Solving ODEs
using Turing,Distributions #Running statistics on MCMC
using CSV, DataFrames #Handle the data
using StatsPlots,ProgressMeter #Plotting and Monitoring

include("HelperFunctions.jl")
include("MCMCRun.jl")

#General workflow for MCMC parameter fitting

#=
1. Set up the ODE
2. Import and process the data
3. Transform the ODE output to match the data
4. Run the MCMC algorithm
5. Postprocessing
=#

###############################################################
                    # 1. Set up the ODE
###############################################################
#Some Model Information needed for MCMC (parameters and States)
parNames = [:k12, :k13,:k14,
            :k21,
            :r31,:k31,
            :k41,:k42,
            :k51,:k52,
            :k61,
            :k71,
            :k81,:k82,
            :k91,
            :k10_1,:k10_2,:k10_3,
            :k11_1,:k11_2,
            :tau1,:tau2,:tau3,:tau4,:tau5,:tau6]

varNames = ["IFN","IFNenv","STAT1","STATP2n","IRF7","IRF7Pn","Target","Eclipse","Productive","Virus"]
#Set the number of parameters and states
parNum = length(parNames)
varNum = length(varNames)

## Define any constants for the model
const k11 = 0.0
const n=3
const TJtot = 0.0001
#Can be defined here or just as a number in the equations

function Model!(dy,y,par,t)
  #IFN, ODE 1 parameters
  #k11=0 #PR8, RIGI is assumed antagonized
  k12=par[1]
  k13=par[2]
  k14=par[3]
  tau1=par[21]
  #IFN_env, ODE 2 parameters
  k21=par[4]
  tau2=par[22]
  #STAT, ODE 3 parameters
  r31=par[5]
  k31=par[6]
  tau3=par[23]
  #STATP, ODE 4 parameters
  k41=par[7]
  k42=par[8]
  tau4=par[24]
  #IRF7, ODE 5 parameters
  k51=par[9]
  k52=par[10]
  tau5=par[25]
  #IRF7P, ODE 6 parameters
  k61=par[11]
  tau6=par[26]
  #Target cells, ODE 7 paramters
  k71=par[12]
  #Eclipse infected cells, ODE 8 parameters
  k81=par[13]
  k82=par[14]
  #Productive infected cells, ODE 9 parameters
  k91=par[15]
  #Viral count, ODE 10 parameters
  k10_1=par[16]
  k10_2=par[17]
  k10_3=par[18]
  #TJ Constants
  #TJ describes the binding of IFN and SOCS feedback
  k11_1=par[19]
  k11_2=par[20]
  TJ=TJtot*(y[2]/(k11_1+y[2])*(1.0/(1.0+k11_2))) #Eq. 11

  #ODE System
  v=y[10]/(10^4) #scale virus effects
  i1=y[8]/(10^4) #scale eclipse cell effects
  dy[1]=(k11*v)+(k12*(v^n))/(k13+(v^n))+k14*y[6]-y[1]*tau1 #IFN in cytoplasm
  dy[2]=(k21*y[1])-(y[2]*tau2) #IFN in environment
  dy[3]=r31+k31*y[4]-y[3]*tau3 #STAT in cytoplasm
  dy[4]=(k41*TJ*y[3])/(k42+y[3])-y[4]*tau4 #STATP
  dy[5]=k51*y[4]+k52*y[6]-y[5]*tau5 #IRF7
  dy[6]=k61*y[5]-y[6]*tau6 #IRF7P
  dy[7]=-k71*y[7]*y[10] #Uninfected target cells
  dy[8]=k71*y[7]*y[10]-(k81*i1)/(1+k82*y[2]) #Eclipse infected cells
  dy[9]=(k81*i1)/(1+k82*y[2])-k91*y[9] #Productive infected cells
  dy[10]=(k10_1*y[9])/(1+k10_2*y[2])-k10_3*y[10] #Virus count

end

#Define information for ODE model
p=rand(parNum) #Parameter values
u0 = [7.94, 0, 262.3, 12.2, 14.15, 0, 0, 250000, 0, 7.5E-2] #Initial Conditions
tspan = (0.25,24.0) #Time (start, end)

#Contruct the ODE Problem
prob = ODEProblem(Model!,u0,tspan,p)
alg = Vern7()  #ODE solver

#This is just me testing the ODEs
sol = solve(prob,alg)
plot(sol,layout=10,legend=false)

###############################################################
                    # 2. Import the data
###############################################################
#Read in the CSV
data = CSV.read("./ViralSensingModel/PR8.csv")

#Convert the data into a workable form
dataTransform = ConvertData(data)


###############################################################
        # 3. Transform the ODE output to match the data
###############################################################


###############################################################
                    # 4. Run the MCMC algorithm
###############################################################

#Provide any prior knowledge for the parameters
priors = fill(FlatPos(0.0),length(prob.p))

#How many MCMC sample do you want
mcmcSamples = 100_000

#Gather all the information to one structure
sampleProblem = MCMCSetup(modelInfo,prob,alg,dataTransform,mcmcSamples,priors)

#Run the MCMC
result = MCMCRun(sampleProblem)

#Save the parameter chains and information about the MCMC run
chainParameters = DataFrame(result,:parameters)
CSV.write("./UserFriendly/Parameters.csv",chainParameters)

chainInternals = DataFrame(result,:internals)
CSV.write("./UserFriendly/Internals.csv",chainInternals)
###############################################################
                    # 5. Postprocessing
###############################################################

#True parameter values
parTure = [0.833, 1.667, 0.167] #Parameter values

#Make a plot of the chains and density plots
chainsPlot = plot(result)
savefig(chainsPlot,"./UserFriendly/ChainsODE.pdf")

#Corner plot for correlations
corPlot = autocorplot(result)
savefig(corPlot,"./UserFriendly/AutoCorr.pdf")

#Running average Plot
runAvePlot = meanplot(result)
savefig(runAvePlot,"./UserFriendly/RunAve.pdf")

#Running average Plot
cornerPlot = corner(result)
savefig(cornerPlot,"./UserFriendly/Corner.pdf")



#Create a plot fitting "best" parameter values
@df data plot(:Time,[:A,:B,:C,:D],markershape=:auto,layout=4,legend=false)

#Retrieve the chains and information about the chains
#Get best parameter set
bestPar = BestParSet(result,parNames) #ignore the std

#Rerun the problem with these parameters
newProb = remake(prob, p=bestPar)
newSol = solve(newProb,alg)

plot!(newSol,layout=4)
savefig("./UserFriendly/DataFit.pdf")
