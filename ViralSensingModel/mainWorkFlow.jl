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

function Model!(dy,y,par,t)
  #IFN, ODE 1 parameters
  k11=0 #PR8, RIGI is assumed antagonized
  k12=par[2]
  n=3
  k13=par[3]
  k14=par[4]
  tau1=par[22]
  #IFN_env, ODE 2 parameters
  k21=par[5]
  tau2=par[23]
  #STAT, ODE 3 parameters
  r31=par[6]
  k31=par[7]
  tau3=par[24]
  #STATP, ODE 4 parameters
  k41=par[8]
  k42=par[9]
  tau4=par[25]
  #IRF7, ODE 5 parameters
  k51=par[10]
  k52=par[11]
  tau5=par[26]
  #IRF7P, ODE 6 parameters
  k61=par[12]
  tau6=par[27]
  #Target cells, ODE 7 paramters
  k71=par[13]
  #Eclipse infected cells, ODE 8 parameters
  k81=par[14]
  k82=par[15]
  #Productive infected cells, ODE 9 parameters
  k91=par[16]
  #Viral count, ODE 10 parameters
  k10_1=par[17]
  k10_2=par[18]
  k10_3=par[19]
  #TJ Constants
  #TJ describes the binding of IFN and SOCS feedback
  TJtot=0.0001
  k11_1=par[20]
  k11_2=par[21]
  TJ=TJtot*(y[2]/(k11_1+y[2])*(1/(1+k11_2))); #Eq. 11

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

###############################################################
                    # 2. Import the data
###############################################################



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