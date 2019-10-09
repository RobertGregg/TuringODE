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

#Some Model Information needed for MCMC
parNames = [:Kab, :Kbc, :Kad]
varNames = [:a, :b, :c,:d]
#Set the number of parameters and states
parNum = length(parNames)
varNum = length(varNames)

modelInfo = ModelInformation(parNames,varNames)

## Define any constants for the model
const V0 = 4/7 #Can be defined here or just as a number in the equations

#Write the rate equations
function Model(du,u,p,t)
  Kab, Kbc, Kad, = p
  a,b,c,d = u

  du[1] = V0*(10.0-a) - Kab*a - Kad*a^2 #a
  du[2] = -V0*b - Kbc*b + Kab*a #b
  du[3] = -V0*c + Kbc*b #c
  du[4] = -V0*d + (1/2)*Kad*a^2 #d
end

#Define Values for ODE model
p = rand(parNum) #Initial arameter values (in order they are defined above)
u0 = zeros(varNum) #Initial Conditions (number of Differential Equations)
tspan = (0.0,10.0) #Time (start, end)

#Contruct the ODE Problem
prob = ODEProblem(Model,u0,tspan,p)
alg = Vern7() #ODE solver

###############################################################
                    # 2. Import the data
###############################################################

#Read in the CSV
data = CSV.read("./UserFriendly/VVdata.csv")

#Convert the data into a workable form
dataTransform = ConvertData(data)

###############################################################
        # 3. Transform the ODE output to match the data
###############################################################


#This function takes in the solved ODE, and outputs the observable species
function ObserveTransform(sol,measuredTime)
  #Create a container to hold the solution at the desired time points
  obversedSpecies = sol(measuredTime)
  #What species are being measured?
  return obversedSpecies
end

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
