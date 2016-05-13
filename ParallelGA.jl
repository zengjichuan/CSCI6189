
module ParallelGA

# -------

importall Base

export  Entity,
        GAmodel,

        runga,
        freeze,
        defrost,
        generation_num,
        population

# -------

abstract Entity

isless(lhs::Entity, rhs::Entity) = lhs.fitness < rhs.fitness

fitness!(ent::Entity, fitness_score) = ent.fitness = fitness_score

# -------

type EntityData
    entity
    generation::Int

    EntityData(entity, generation::Int) = new(entity, generation)
    EntityData(entity, model) = new(entity, model.gen_num)
end

# -------

type Population
  initial_pop_size::Int
  gen_num::Int

  ga_model::GAmodel
end

function iterate(P::Population)
  # iterate over generation
  # create light weight thread to compute fitness
  grouper = @task P.ga_model.ga.group_entities(model.population)
  groupings = Any[]
  while !istaskdone(grouper)
      group = consume(grouper)
      group != nothing && push!(groupings, group)
  end

  if length(groupings) < 1
      break
  end

  crossover_population(P.ga_model, groupings)
  mutate_population(P.ga_model)
end


type GAmodel


    islandCount::Int
    islands::Array{Any,1}
    maxGenerations::Int
    numMigrate::Int
    migrationRate::Float64
    tolerance::Float64

    population::Array
    pop_data::Array{EntityData}
    freezer::Array{EntityData}

    rng::AbstractRNG

    ga

    GAmodel() = new(0, 1, 1, Any[], 0, 0.0, Any[], EntityData[], EntityData[], MersenneTwister(time_ns()), nothing)
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


function runga(mdl::Module; initial_pop_size = 128)
  if mdl.islandCount > 1    # need to define islandCount in test Module
    n = nprocs()
    islandCount = min(n-1, mdl.islandCount)
  end

  model = GAmodel()
  model.ga = mdl
  model.initial_pop_size = initial_pop_size

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
        put!(Rs[myid()], success)
      end
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

function runga(model::GAmodel)
    reset_model(model)
    create_initial_population(model)

    while true
        evaluate_population(model)

        grouper = @task model.ga.group_entities(model.population)
        groupings = Any[]
        while !istaskdone(grouper)
            group = consume(grouper)
            group != nothing && push!(groupings, group)
        end

        if length(groupings) < 1
            break
        end

        crossover_population(model, groupings)
        mutate_population(model)
    end

    model
end

# -------

function reset_model(model::GAmodel)
    global _g_model = model

    model.gen_num = 1
    empty!(model.population)
    empty!(model.pop_data)
    empty!(model.freezer)
end

function create_initial_population(model::GAmodel)
    for i = 1:model.initial_pop_size
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
    old_pop = model.population

    model.population = Any[]
    sizehint(model.population, length(old_pop))

    model.pop_data = EntityData[]
    sizehint(model.pop_data, length(old_pop))

    model.gen_num += 1

    for group in groupings
        parents = { old_pop[i] for i in group }
        entity = model.ga.crossover(parents)

        push!(model.population, entity)
        push!(model.pop_data, EntityData(model.ga.crossover(parents), model.gen_num))
    end
end

function mutate_population(model::GAmodel)
    for entity in model.population
        model.ga.mutate(entity)
    end
end

function getGlobalParetoFront(model::GAmodel)
  if model.islandCount == 1
    globalParetoFront = model.islands[1].simplifiedParetoFront
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
