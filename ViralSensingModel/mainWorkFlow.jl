#Importing Neccessary Packages
using DifferentialEquations,ModelingToolkit #Solving ODEs
using Turing,Distributions #Running statistics on MCMC
using CSV, DataFrames #Handle the data
using StatsPlots,ProgressMeter #Plotting and Monitoring

#Not sure why but include assumes you're in same folder as file(?)
include("../HelperFunctions.jl")
include("../MCMCRun.jl")

if pwd()!="C:\\Users\\Portable\\Documents\\GitHub\\TuringODE\\ViralSensingModel"
  cd("./ViralSensingModel")
end


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
            :k21, :tau2,
            :k31,:k32,
            :k41,:k42,
            :k51,
            :k61,
            :k71, :k72,
            :k81,:k82]

varNames = ["IFN","IFNe","STATP2n","IRF7","IRF7P","Infected","Virus"]
#Set the number of parameters and states
parNum = length(parNames)
varNum = length(varNames)
modelInfo = ModelInformation(parNames,varNames)

## Define any constants for the model
const k11 = 0.0
const n=3
const JAKtot = 0.0001
const tau3 = 1.238 #Cambridge 2011
const tau4 = 0.347 #Sharova 2009
const tau5 = 6.93 #Prakash 2006
#Can be defined here or just as a number in the equations

function Model!(dy,y,par,t)
  #IFN, ODE 1 parameters
  #k11=0 #PR8, RIGI is assumed antagonized
  k12=par[1]
  k13=par[2]
  k14=par[3]
  #IFNe, ODE 2 parameters
  k21=par[4]
  tau2=par[5]
  #STATP, ODE 3 parameters
  k31=par[6]
  k32=par[7]
  #IRF7, ODE 4 parameters
  k41=par[8]
  k42=par[9]
  #IRF7P, ODE 5 parameters
  k51=par[10]
  #Infected cells, ODE 6 parameters
  k61=par[11]
  #Viral replication, ODE 7 paramters
  k71=par[12]
  k72=par[13]
  #TJ Constants
  #TJ describes the binding of IFN and SOCS feedback
  k81=par[14]
  k82=par[15]
  JAK=JAKtot*(y[2]/(k81+y[2])*(1.0/(1.0+k82))) #Eq. 11

  #ODE System
  v=y[7]/(10^4) #scale virus effects
  dy[1]=k11*v+(k12*(v^n))/(k13+(v^n))+k14*y[5]-k21*y[1] #IFN in cytoplasm
  dy[2]=k21*y[1]-y[2]*tau2 #IFN in environment
  dy[3]=(k31*JAK)/(k32+JAK)-y[3]*tau3 #STATP
  dy[4]=k41*y[3]+k42*y[5]-y[4]*tau4 #IRF7
  dy[5]=k51*y[4]-y[5]*tau5 #IRF7P
  dy[6]=-k61*y[6] #Productive infected cells
  dy[7]=(k71*y[6]*(v^n))/(y[2]*(v^n))-k72*y[7] #Virus count

end

#Define information for ODE model
p=rand(parNum) #Parameter values
u0 = [7.94, 0.01, 12.2, 14.15, 0.01, 250000, 7.5E-2] #Initial Conditions
tspan = (0.25,24.0) #Time (start, end)

#Contruct the ODE Problem
prob = ODEProblem(Model!,u0,tspan,p)
alg = Rodas5()  #ODE solver

#Testing the ODEs
sol = solve(prob,alg)
plot(sol,layout=varNum,legend=false, framestyle=:box,title=[v for i=1:1, v in varNames])
xlabel!("")

###############################################################
                    # 2. Import the data
###############################################################
#Read in the CSV
data = CSV.read("PR8.csv", missingstring= "-")

#What if we normalize the virus?
#vMin = 7.5E-2 #Initial Condition
#vMax = maximum(skipmissing(data.Virus))
#virusNorm(x) = @. (x - vMin)/(vMax - vMin)
#data.Virus = virusNorm(data.Virus)

control = CSV.read("Control.csv", missingstring= "-")

#Convert the data into a workable form
dataTransform = ConvertData(data)


###############################################################
        # 3. Transform the ODE output to match the data
###############################################################

#What states are being measured?
const allMeasuredStates = [1,3,4,7] #IFN, STATp, IRF7, Virus

#This function takes in the solved ODE, and outputs the observable species
function ObserveTransform(sol,measuredTime,currentStates)
  #Create a container to hold the solution at the desired time points
  #Get indices of currently measured states
  measuredIdx = allMeasuredStates[currentStates]
  obversedSpecies = sol(measuredTime)[measuredIdx]

  #Create a vector of control data
  timeFilter = x->x==measuredTime

if 7 ∉ measuredIdx #if not virus
  #Get the control values and calculate a lfc
  controlVec = convert(Vector,control[findfirst(timeFilter,control.Time),3:end])
  obversedSpecies = @. log2(maximum(obversedSpecies,0.0)+1.0) - log2(controlVec)

elseif 7 ∈ measuredIdx && length(measuredIdx)==1 #Only Virus measured
  obversedSpecies = obversedSpecies
else #Both virus and species are measured
  #Get the control values and calculate a lfc, but skip the virus
  controlVec = convert(Vector,control[findfirst(timeFilter,control.Time),3:end])
  obversedSpecies[1:end-1] = @. log2(max(obversedSpecies[1:end-1],0.0)+1.0) - log2(controlVec)
end

  return obversedSpecies
end

###############################################################
                    # 4. Run the MCMC algorithm
###############################################################

#Provide any prior knowledge for the parameters
#priors = fill(FlatPos(0.0),length(prob.p))
priors = fill(Uniform(0.0,100.0),length(prob.p))
priors[5]=Uniform(0,10) #tau2, IFNe degradation
priors[11]=Uniform(0,1) #k61 Infected cell clearance
priors[13]=Uniform(0,1) #k72, viral clearance
#How many MCMC sample do you want
mcmcSamples = 10

#Gather all the information to one structure
sampleProblem = MCMCSetup(modelInfo,prob,alg,dataTransform,mcmcSamples,priors)

#Run the MCMC
result = MCMCRun(sampleProblem)

#Save the parameter chains and information about the MCMC run
chainParameters = DataFrame(result,:parameters)
CSV.write("./Parameters.csv",chainParameters)

chainInternals = DataFrame(result,:internals)
CSV.write("./Internals.csv",chainInternals)
###############################################################
                    # 5. Postprocessing
###############################################################

#Make a plot of the chains and density plots
chainsPlot = plot(result)
savefig(chainsPlot,"./Figures/ChainsODE.pdf")

#Corner plot for correlations
#corPlot = autocorplot(result)
#savefig(corPlot,"./Figures/AutoCorr.pdf")

#Running average Plot
runAvePlot = meanplot(result)
savefig(runAvePlot,"./Figures/RunAve.pdf")

#Running average Plot
#cornerPlot = corner(result)
#savefig(cornerPlot,"./Figures/Corner.pdf")



#Create a plot fitting "best" parameter values
@df data plot(:Time,[:IFN,:STATP,:IRF7,:Virus],markershape=:auto,
              layout=length(allMeasuredStates),legend=false)

#Retrieve the chains and information about the chains
#Get best parameter set
bestPar = BestParSet(result,parNames) #ignore the std


#Rerun the problem with these parameters
newProb = remake(prob, p=bestPar)
newSol = solve(newProb,alg)

plotBestSol = zeros(length(newSol.t),varNum)
controlLFC = log2.(mean(convert(Matrix,control[:,3:end]),dims=1))

for t=1:length(newSol.t)
  for s = 1:varNum
    if s ∈ allMeasuredStates[1:end-1]
      idx = findfirst(x->x==s,allMeasuredStates)
      #@show s,t,idx
      plotBestSol[t,s] = log2(max(newSol[s,t],0.0)+1.0) - controlLFC[idx]
    else
      plotBestSol[t,s] = newSol[s,t]
    end
  end
end

plot!(newSol.t,plotBestSol[:,allMeasuredStates],layout=length(allMeasuredStates),legend=false,title=[v for i=1:1, v in varNames[allMeasuredStates]])
plot!(newSol,vars=allMeasuredStates,layout=5)
savefig("./Figures/DataFit.pdf")


plot(newSol,layout=varNum,legend=false, framestyle=:box,title=[v for i=1:1, v in varNames])
xlabel!("")
savefig("./Figures/AllODEs.pdf")
