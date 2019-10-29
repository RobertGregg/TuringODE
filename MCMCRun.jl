
function MCMCRun(Problem::MCMCSetup)

#Assume additive Guassian noise for each measurement
likelihood = (μ,σ) -> MvNormal(μ,σ*ones(length(μ)))
#Choose an MCMC sampler
sampler = Turing.NUTS(0.65) #0.65 is the desired acceptance ratio
#Define all of the parameters
parTuple = Problem.MI.Parameters[end]==:σ ? Tuple{(Problem.MI.Parameters...)} : Tuple{(push!(Problem.MI.Parameters,:σ))...} #Tack on the std
# Define the default value when no data is passed
allData = [experiment[2] for experiment in Problem.Data]
defaults = (x = allData,)

modelFramework(varInfo,sampler,model) = begin
  #Set the accumulated log Posterior to zero
  varInfo.logp = 0.0
  #Set the default data for when no input is given
  x = model.defaults.x

  #Loop through the priors and assign them to the model
  θ = Vector(undef,length(Problem.Priors))

  for (prior,name,i) in zip(Problem.Priors,parNames,1:length(Problem.MI.Parameters))
    θ[i], logp = Turing.assume(sampler,prior,Turing.VarName([:mf,name],""),varInfo)
    varInfo.logp += logp
  end

  #Convert θ to array
  θ = convert(Array{typeof(first(θ))},θ) #Is this needed?

  #Assume normal random error in data
  likelihoodDistPriors = InverseGamma(2,3) #prior for data? Why this in particular?
  σ, logp = Turing.assume(sampler,likelihoodDistPriors,Turing.VarName([:mf,parNames[end]],""),varInfo)
  varInfo.logp += logp

  #Remake a new problem with the different parameter set
  newProb = remake(Problem.ODE,u0 = convert.(eltype(θ),(Problem.ODE.u0)), p=θ)
  sol = solve(newProb,Problem.Alg)

  #Need to loop through experiments to compare to model
  for (times,data) in Problem.Data

    if sol.retcode != :Success
      #println(newp) #what were the bad pars?
      varInfo.logp -= Inf
    else
      #Loop through time
      for (idx,t) in enumerate(times)
        #What states are being measured?
          measured = @. !ismissing(data[idx,:])
        #What are their concentrations?
          obs = ObserveTransform(sol,t,measured)
        #Update the likelihood with each measurement
        logp = Turing.observe(
          sampler,
          likelihood(obs,σ),
          data[idx,measured],
          varInfo)

        varInfo.logp += logp
      end #Time points
    end #If solve succeeds
  end #Loop through data

  return varInfo
end #modelFramework


# Instantiate a Model object.
model = Turing.Model{parTuple, Tuple{:x}}(modelFramework, allData, defaults)
chain = sample(model,sampler,Problem.Samples)
end #mcmcODE function
