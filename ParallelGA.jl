
module ParallelGA

# -------

importall Base

export  Entity,
        GAmodel,

        runga

# -------

abstract Entity


# -------

type Population
  pop_data::Array{Entity, 1}
  pop_size::Int
  gen_num::Int
  paretoFront::Array{Entity, 1}


  ga_model

  function Population(pop_size::Int, ga_model)
    this = new()
    this.ga_model = ga_model
    this.pop_size = pop_size
    this.gen_num = 0
    # initialize population
    this.pop_data = Entity[]
    for i = 1:this.pop_size
      push!(this.pop_data, Entity(model.ga.create_entity(i)))
    end

    calcPareto(this, P.pop_data)
    return this
  end
end

function calcPareto(P::Population, newEntities)
  fit = pmap(P.ga_model.fitness, newEntities)   # need to require
  for (ind, ent) in enumerate(newEntities)
    # assess entity for fitness
    ent.fitness = fit[ind]

    if !isnan(ent.fitness) && ent.fitness < Inf
      if length(P.paretoFront) == 0
        push!(P.paretoFront, ent)
      else
        dominated = false
        index = 1
        while index <= length(P.paretoFront)
          paretoEntity = P.paretoFront[index]
          if ent.dominates(paretoEntity)
            # remove pareto front
            splice!(P.paretoFront, index)
          elseif paretoEntity.dominates(ent)
            dominated = true
            break
          else
            index += 1
          end
          if !dominated
            push!(P.paretoFront, ent)
          end
        end
      end
    end
  end
end

function simplifiedParetoFront(P::Population)
# sort the paretoFront according to the one of the fitness function
# here we use the number of non-zero features as the second dimention
  if length(P.paretoFront) == 0
    return []
  end
  nodeCounts = [t.numNodes for t in P.paretoFront]
  m = minimum(nodeCounts)
  M = minimum(nodeCounts)
  lastFit = Inf
  simpleParetoFront = []
  for i = m:M
    temp = []
    for t in P.paretoFront
      if t.numNodes == i
        push!(temp, t)
      end
    end
    if length(temp) > 0
      minFitEntity = temp[1]
      for t in temp[2:end]
        if t.fitness < minFitEntity.fitness
          minFitEntity = t
        end
      end
      # only one entity per length, following the defination of paretofront
      if minFitEntity.fitness <= lastFit
        push!(simpleParetoFront, minFitEntity)
        lastFit = minFitEntity.fitness
      end
    end
  end
  return simpleParetoFront
end

function iterate(P::Population)
  # iterate over generation

  # select parents
  grouper = @task P.ga_model.ga.group_entities(P)
  groupings = Any[]
  while !istaskdone(grouper)
      group = consume(grouper)
      group != nothing && push!(groupings, group)
  end

  if length(groupings) < 1
      println("no pretofront!")
  end

  P.pop_data = EntityData[]
  paretoFrontSize = length(P.paretoFront)
  # keep all the pareto front
  append!(P.pop_data, P.paretoFront)
  # # just keep 2 elite children
  # append!(model.population, model.population.paretoFront[[rand(1:paretoFrontSize), rand(1:paretoFrontSize)]])
  newGen = EntityData[]

  for r in rand(1, model.popSize - paretoFrontSize)
    # do crossover
    push!(newGen, EntityData(model.ga.crossover(groupings)))
    # do mutation
    if r < P.ga_model.mutationRate
      model.ga.mutate(newGen[end])
    end
  end
  append!(P.pop_data, newGen)
  calcPareto(P, newGen)

end


type GAmodel

    islandCount::Int
    islands::Array{Any,1}
    popSize::Int
    maxGenerations::Int
    numMigrate::Int
    migrationRate::Float64

    tolerance::Float64

    # population::Array
    # pop_data::Array{EntityData}
    # freezer::Array{EntityData}
    #
    # rng::AbstractRNG

    ga

    function GAmodel(islandCount=1, popSize=100, maxGenerations=100, numMigrate=0.1, migrationRate=2, tol=.1)
      this = new()
      this.islandCount = islandCount
      this.popSize = popSize
      this.maxGenerations = maxGenerations
      this.islands = Any[]

      this.numMigrate = numMigrate
      this.migrationRate = migrationRate

      this.tolerance = tol
      return this
    end

end

global _g_model

# -------

function migrate(model::GAmodel)
  @everywhere begin
    temp = EntityData[]

    numPrareto = length(P.paretoFont)
    if numPareto > 0
      for i = 1:model.numMigrate
        push!(temp, P.paretoFont[rand(1:numPareto)])
      end
    end
    # move entity to the next island
    put!(Rs[((myid()+1) % nprocs())+1],temp)
  end

  @everywhere begin
    temp = take!(Rs[myid()])
    for entity in temp
      push!(P.paretoFront, entity)
    end
  end
end

function freeze(model::GAmodel, entity::EntityData)
    push!(model.freezer, entity)
    println("Freezing: ", entity)
end

function freeze(model::GAmodel, entity)
    entitydata = EntityData(entity, model.gen_num)
    freeze(model, entitydata)
end

freeze(entity) = freeze(_g_model, entity)


function defrost(model::GAmodel, generation::Int)
    filter(model.freezer) do entitydata
        entitydata.generation == generation
    end
end

defrost(generation::Int) = defrost(_g_model, generation)


generation_num(model::GAmodel = _g_model) = model.gen_num


population(model::GAmodel = _g_model) = model.population


function runga(mdl::Module; islandCount=1, popSize=100, maxGenerations=100, numMigrate=0.1, migrationRate=2, tol=.1)
  if islandCount > 1    # need to define islandCount in test Module
    n = nprocs()
    islandCount = min(n-1, islandCount)
  end

  model = GAmodel(islandCount, popSize, maxGenerations, numMigrate, migrationRate, tol)
  model.ga = mdl

  numGen = 0

  if islandCount > 1
    println("Running in parallel")
    # @everywhere begin
    #   model = GAmodel()
    #   ...
    # end
    # Julia seems unable to serialized user-defined dataypes, so
    # we need to accomplish this by communicating only Julia natives
    #
    r = RemoteRef()

    put!(r, model)
    @sync for p in procs()
      @spawnat p eval(Main, Expr(:(=),:model, fetch(r)))
    end
    take!(r)

    @everywhere begin
      P = Population(model.initial_pop_size, 0, model)
    end
    println("Populations initialized")

    goodEnough = false
    r = RemoteRef()
    num = round(Int, model.maxGenerations * model.migrationRate)
    numToIterate = model.maxGenerations / num

    put!(r, num)
    @sync for p in procs()
        @spawnat p eval(Main,Expr(:(=),:numToIterate,fetch(r)))
    end
    take!(r)

    for I = 1:num
      @everywhere begin
        for i = 1:numToIterate
          P.iterate()
        end
        success = false
        for can in P.paretoFront
          if  can.fitness < model.tolerance
            success = true
            break
          end
        end
        # update success state
        put!(Rs[myid()], success)
      end
      # if success is in local island
      for r in Rs
        if take!(r)
          goodEnough = true
        end
      end
      numGen += numToIterate
      println(numGen)
      if goodEnough
        println("Perfect Solution Found!")
        break
      else
        migrate(model)
      end
    end
    num = model.maxGenerations - num * numToIterate
    if num > 0 && !goodEnough
      put!(r,num)
      @sync for p in procs()
          @spawnat p eval(Main,Expr(:(=),:numToIterate,fetch(r)))
      end
      take!(r)

      @everywhere begin
          for i in 1:numToIterate
              P.iterate()
          end
      end
      numGen += num
    end
  else
    println("Running is serial")
    push!(model.islands, Population(model.initial_pop_size, 0, model))
    P = model.islands[1]
    for i = 1:model.maxGenerations
      P.iterate()
    end
    numGen += model.maxGenerations
  end

  return getGlobalParetoFront(model)
end

# function runga(model::GAmodel)
#     reset_model(model)
#     create_initial_population(model)
#
#     while true
#         evaluate_population(model)
#
#         grouper = @task model.ga.group_entities(model.population)
#         groupings = Any[]
#         while !istaskdone(grouper)
#             group = consume(grouper)
#             group != nothing && push!(groupings, group)
#         end
#
#         if length(groupings) < 1
#             break
#         end
#
#         crossover_population(model, groupings)
#         mutate_population(model)
#     end
#
#     model
# end

# -------

function reset_model(model::GAmodel)
    global _g_model = model

    model.gen_num = 1
    empty!(model.population)
    empty!(model.pop_data)
    empty!(model.freezer)
end

function create_initial_population(model::GAmodel)
    for i = 1:model.popSize
        entity = model.ga.create_entity(i)

        push!(model.population, entity)
        push!(model.pop_data, EntityData(entity, model.gen_num))
    end
end

function evaluate_population(model::GAmodel)
    scores = pmap(model.ga.fitness, model.population)
    for i in 1:length(scores)
        fitness!(model.population[i], scores[i])
    end

    sort!(model.population; rev = true)
end

function crossover_population(model::GAmodel, groupings)

    model.population = Any[]
    paretoFrontSize = length(model.population.paretoFront)
    # keep all the pareto front
    append!(model.population, model.population.paretoFront)
    # # just keep 2 elite children
    # append!(model.population, model.population.paretoFront[[rand(1:paretoFrontSize), rand(1:paretoFrontSize)]])

    for r = 1:(model.popSize - paretoFrontSize)
      push!(model.population, model.ga.crossover(groupings))
    end

    # old_pop = model.population
    #
    # model.population = Any[]
    # sizehint(model.population, length(old_pop))
    #
    # model.pop_data = EntityData[]
    # sizehint(model.pop_data, length(old_pop))
    #
    # model.gen_num += 1
    #
    # for group in groupings
    #     parents = { old_pop[i] for i in group }
    #     entity = model.ga.crossover(parents)
    #
    #     push!(model.population, entity)
    #     push!(model.pop_data, EntityData(model.ga.crossover(parents), model.gen_num))
    # end
end

function mutate_population(model::GAmodel)
    for entity in model.population
        model.ga.mutate(entity)
    end
end

function getGlobalParetoFront(model::GAmodel)
  if model.islandCount == 1
    globalParetoFront = simplifiedParetoFront(model.islands[1])
  else
    globalParetoFront = EntityData[]
    cans = EntityData[]
    PF = EntityData[]
    @sync for p in procs()
      @spawnat p put!(Rs[myid()],P.paretoFront);
    end
    for r in Rs
      append!(cans,take!(r));
    end
    for can in cans
      if length(globalParetoFront) == 0
        push!(PF, can)
      else
        # For each entity already in the pareto front,
        # make sure the pareto entity has lower fitness and
        # sparsity.
        dominated = false
        index = 1
        while index <= length(PF)
          paretoEntity = PF[index]
          if can.domiantes(paretoEntity)
            # remove pareto entity
            splice!(PF, index)
          elseif paretoEntity.dominates(can)
            # current entity is dominated
            dominated = true
            break
          elseif can.fitness == paretoEntity.fitness &&
            can.numNodes == paretoEntity.numNodes
            dominated = true
            break
          else
            index += 1
          end
        end
        if !dominated
          push!(PF, can)
        end
      end
    end
    if length(PF) == 0
      globalParetoFront = EntityData[]
      return EntityData[]
    end
    nodeCounts = [t.numNodes for t in PF]
    m = minimum(nodeCounts)
    M = maximum(nodeCounts)
    lastFit = Inf
    simpleParetoFront = []
    for i in m:M
        temp = []
        for t in PF
            if t.numNodes == i
                push!(temp,t)
            end
        end
        if length(temp) > 0
            minFitEntity = temp[1]
            for t in temp[2:end]
                if t.fitness < minFitEntity.fitness
                    minFitEntity = t
                end
            end
            if minFitEntity.fitness <= lastFit
                push!(simpleParetoFront,minFitEntity)
                lastFit = minFitEntity.fitness
            end
        end
    end
    globalParetoFront = simpleParetoFront
  end
  return globalParetoFront
end


end
