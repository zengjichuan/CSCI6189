module testPGA

using ParallelGA

type TestEntity <: Entity
  abcde::Array
  numNodes::Int
  fitness::Float64
  dominates::Function
  function TestEntity(abcde::Array)
    this = new()
    this.abcde = abcde
    numNodes = countnz(abcde)
    this.fitness = -1
    this.dominates = function(o_entity::TestEntity)
      if this.fitness <= o_entity.fitness && this.numNodes <= o_entity.numNodes
        if this.fitness != o_entity.fitness || this.numNodes != o_entity.numNodes
          return true
        end
        return false
      end
      return false
    end
    return this
  end
end

function create_entity(num)
  TestEntity(rand(Int, 5) % 43)
end

function fitness(ent)
  score = ent.abcde[1] + 2 * ent.abcde[2] + 3 * ent.abcde[3] + 4 * ent.abcde[4] + 5 * ent.abcde[5]
  abs(score -42)
end

function group_entities(pop)
  for i in pop.paretoFront
    produce(i)
  end
end

function crossover(group)
  child = []
  count = length(group)
  sizehint!(child, length(group[1].abcde))

  parent1 = group[rand(1:count)]
  parent2 = group[rand(1:count)]

  for i =1:length(group[1].abcde)
    if rand(1:count)%2 == 0
      push!(child, parent1.abcde[i])
    else
      push!(child, parent2.abcde[i])
    end
  end
  TestEntity(child)
end

function mutate(ent)
  count = length(ent.abcde)
  ent.abcde[rand(1:count)] = rand(Int) % 43
end

function test_serial()
  runga(1, 100, 100, 0.1, 10, .1)
end

function test_parallel(nprocs_to_add = 2)
  addprocs(nprocs_to_add)
  require("testPGA.jl")
  println("nprocs: $(nprocs())")
  runga(1, 100, 100, 0.1, 10, .1)
end
end
