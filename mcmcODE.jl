function mcmcODE(prob::DiffEqBase.DEProblem,alg,t,data,priors,parVary,parVaryIdx;
  likelihood = (μ,σ) -> MvNormal(μ,σ*ones(length(μ))),
  num_samples=1000,
  sampler = Turing.NUTS(num_samples,0.65)) #0.65 is the desired acceptance ratio

  modelFramework(varInfo,sampler,model) = begin

  #Set the accumulated log Posterior to zero
  varInfo.logp = 0.0

  #Set the default data for when no input is given
  x = model.defaults.x

  #Loop through the priors and assign them to the model
  θ = Vector(undef,length(priors))

  for ((i,prior),parName) in zip(enumerate(priors),parVary)
    θ[i], logp = Turing.assume(sampler,prior,Turing.VarName([:mf,parName],""),varInfo)

    varInfo.logp += logp
  end

  #Convert θ to array
  θ = convert(Array{typeof(first(θ))},θ) #Is this needed?

  #Assume normal random error in data
  likelihoodDistPriors = InverseGamma(2,3) #prior for data? what is σ?
  σ, logp = Turing.assume(sampler,likelihoodDistPriors,Turing.VarName([:mf,:σ],""),varInfo)
  varInfo.logp += logp

  #Remake a new problem with the different parameter set
  newp = convert.(eltype(θ),(prob.p))
  newp[parVaryIdx] = θ
  newProb = remake(prob,u0 = convert.(eltype(θ),(prob.u0)), p=newp)
  sol = solve(newProb,alg)

  #Need to loop through experiments
  for (times,data) in zip(timePoints,experiments)

    if sol.retcode != :Success
      #println(newp) #what were the bad pars?
      varInfo.logp -= Inf
    else
      #What states are being measured?
      expIdx = @. !ismissing(data[1,:])

      #Loop through time
      #What are their concentrations?
      allObs = Observables(sol,intConditions,dataIndex,times)
      obs = vec(allObs[expIdx,:])

      logp = Turing.observe(
        sampler,
        likelihood(obs,σ),
        data[:,expIdx],
        varInfo)

      varInfo.logp += sum(logp)

      #Loop through time points
      #=
      for (i,t) in enumerate(times)
        logp = Turing.observe(
          sampler,
          likelihood(obs,σ),
          data[i,expIdx],
          varInfo)

        varInfo.logp += logp
      end #loop time
      =#

    end #successful integration
  end #experiments

#Heuristics
varInfo.logp += heuristic(sol,desiredScale)

    return varInfo

  end #Model

#Define all of the parameters

parTuple = Tuple{([collect(keys(parChange)); [:σ]])...}

# Define the default value for x when missing
defaults = (x = data,)

# Instantiate a Model object.
model = Turing.Model{parTuple, Tuple{:x}}(modelFramework, data, defaults)

chain = sample(model,sampler)
end #mcmcODE function
