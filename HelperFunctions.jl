#This is a structure that hold all of the information needed about the Model

mutable struct ModelInformation
  Parameters::Array{Symbol,1}
  States::Array{Symbol,1}
end

mutable struct MCMCSetup
  MI::ModelInformation
  ODE::ODEProblem
  Alg::OrdinaryDiffEqAlgorithm
  Data::Array
  Samples::Int64
  Priors::Array{UnivariateDistribution,1}
end


function ConvertData(data::DataFrame)
  #Convert Data into an array of arrays for iterating
  timePoints = [data[data.Experiment .==i,2] for i ∈ unique(data.Experiment)]
  experiments = [data[data.Experiment .==i,3:end] for i ∈ unique(data.Experiment)]
  #Convert into matrices and vectors
  timeVec = convert.(Vector,timePoints)
  expArray = convert.(Matrix,experiments)

  return [i for i in zip(timeVec,expArray)]
end



function BestParSet(result::Chains,parNames)
  m = mean(result)
  bestParTup = NamedTuple{tuple(m[:parameters]...)}(m[:mean])

  return [bestParTup[p] for p in parNames]
end
